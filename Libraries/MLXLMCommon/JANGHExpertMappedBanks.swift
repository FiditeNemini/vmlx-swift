import Cmlx
import Foundation
import MLX
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Experimental decode owner: full packed banks never become Metal resources here.
/// The cache cap bounds this owner's retained views, not in-flight references or
/// process/host memory. It does not change any user allocator or residency limit.
final class JANGHExpertMappedBanks {
    enum Storage: Equatable { case independentMappings, wholeBankViews }
    let identity = UUID()
    var contract: JANGHFormatContract { source.contract }

    struct Selection {
        let ownerIdentity: UUID
        let module: String
        let expertIDs: [UInt32]
        /// Ready [1, N, Kwords] arrays in route order; duplicate IDs reuse one object.
        let packed: [MLXArray]
        /// Small full [E, N] scale table, indexed by original expert ID in the kernel.
        let scales: MLXArray
        /// Unique packed mapped spans in this selection, including page prefixes.
        let mappedPackedBytes: Int
        fileprivate init(ownerIdentity: UUID, module: String, expertIDs: [UInt32], packed: [MLXArray],
                         scales: MLXArray, mappedPackedBytes: Int) {
            self.ownerIdentity = ownerIdentity
            self.module = module
            self.expertIDs = expertIDs
            self.packed = packed
            self.scales = scales
            self.mappedPackedBytes = mappedPackedBytes
        }
    }

    private struct Key: Hashable {
        let module: String
        /// nil is the full scale table; nonnil is one packed expert.
        let expert: UInt32?
    }
    private struct Entry {
        let array: MLXArray
        let mappedBytes: Int
    }
    private let source: JANGHMappedBanks.SourceLease
    private let cacheByteLimit: Int
    // Explicit causal diagnostic only: this owner retains full bank resources.
    // The view-cache cap does not bound this additional whole-bank allocation.
    private let diagnosticWholeBanks: JANGHMappedBanks?
    private var diagnosticExpertViews: [String: [MLXArray]] = [:]
    private let lock = NSLock()
    private var entries: [Key: Entry] = [:]
    private var leastRecentFirst: [Key] = []
    private var retainedBytes = 0
    private var selections = 0
    private var cacheHits = 0
    private var cacheMisses = 0
    /// Checkpoint bytes, regardless of how many GPU views have been materialized.
    let logicalCheckpointBytes: Int

    init(source: JANGHMappedBanks.SourceLease, cacheByteLimit: Int,
         storage: Storage = .independentMappings) throws {
        guard cacheByteLimit >= 0 else {
            throw JANGHFormatContract.ValidationError.invalid("negative JANGH expert view cache cap")
        }
        var total = 0
        for pair in source.plan.projections.values {
            for location in [pair.packed, pair.scales] {
                let sum = total.addingReportingOverflow(location.byteCount)
                guard !sum.overflow else {
                    throw JANGHFormatContract.ValidationError.invalid("JANGH logical byte count overflow")
                }
                total = sum.partialValue
                // Metadata/descriptor check only: no mapped buffer or payload read.
                try source.withValidatedFile(for: location) { _, _ in }
            }
        }
        self.source = source
        self.cacheByteLimit = cacheByteLimit
        self.logicalCheckpointBytes = total
        diagnosticWholeBanks = storage == .wholeBankViews ? try JANGHMappedBanks(source: source) : nil
        if let whole = diagnosticWholeBanks {
            for module in source.plan.projections.keys.sorted() {
                let bank = try whole.projection(module).packed
                let views = (0..<bank.dim(0)).map { bank[$0..<($0 + 1)] }
                // Materialize metadata at load, not a new eval boundary per route.
                // Core Slice aliases the ready bank without a payload allocation.
                eval(views)
                for view in views {
                    try JANGHBankLayout.requireReadyRowContiguous(view, role: "whole-bank expert view")
                }
                diagnosticExpertViews[module] = views
            }
            print("JANGH_SELECTED_STORAGE whole_bank_views logical_bytes=\(total); full bank resources retained, cache cap bounds view entries only")
        }
    }

    func isBacked(by candidate: JANGHMappedBanks.SourceLease) -> Bool { source === candidate }

    /// Owner-retained GPU allocation spans, not global active bytes. Caller-held
    /// selections and queued Metal work can keep evicted views alive separately.
    var cachedOwnedMappedBytes: Int {
        lock.lock()
        defer { lock.unlock() }
        return retainedBytes
    }

    func removeAllCachedViews() {
        lock.lock()
        entries.removeAll()
        leastRecentFirst.removeAll()
        retainedBytes = 0
        lock.unlock()
    }

    /// B1/top8 diagnostic admission. No hidden full-bank fallback or GPU-index
    /// readback: the caller explicitly supplies the measured router's host IDs.
    func selection(module: String, expertIDs: [UInt32]) throws -> Selection {
        guard expertIDs.count == 8, let pair = source.plan.projections[module] else {
            throw JANGHFormatContract.ValidationError.invalid("JANGH expert selection requires a known module and eight routes")
        }
        lock.lock()
        defer { lock.unlock() }
        selections += 1
        defer {
            if selections.isMultiple(of: 4096) {
                print("JANGH_SELECTED_CACHE selections=\(selections) hits=\(cacheHits) misses=\(cacheMisses) retained_bytes=\(retainedBytes) cap_bytes=\(cacheByteLimit)")
            }
        }
        return try source.withValidatedFile(for: pair.packed) { packedFD, packedHeader in
            try source.withValidatedFile(for: pair.scales) { scaleFD, scaleHeader in
                guard packedHeader.dtype == "U32", packedHeader.shape.count == 3,
                    scaleHeader.dtype == "F16", scaleHeader.shape.count == 2,
                    scaleHeader.shape == Array(packedHeader.shape.prefix(2)),
                    expertIDs.allSatisfy({ UInt64($0) < UInt64(packedHeader.shape[0]) })
                else {
                    throw JANGHFormatContract.ValidationError.invalid("invalid JANGH selected expert geometry")
                }
                let experts = packedHeader.shape[0]
                guard pair.packed.byteCount % experts == 0 else {
                    throw JANGHFormatContract.ValidationError.invalid("unaligned JANGH packed expert span")
                }
                let perExpert = pair.packed.byteCount / experts
                let scales = try cached(Key(module: module, expert: nil)) {
                    if let whole = diagnosticWholeBanks {
                        return Entry(array: try whole.projection(module).scales,
                                     mappedBytes: pair.scales.byteCount)
                    }
                    return try map(fd: scaleFD, offset: pair.scales.fileOffset,
                            bytes: pair.scales.byteCount, shape: scaleHeader.shape, dtype: .float16)
                }
                var unique: [UInt32: Entry] = [:]
                var packed: [MLXArray] = []
                var mappedBytes = 0
                for expert in expertIDs {
                    let entry: Entry
                    if let same = unique[expert] {
                        entry = same
                    } else {
                        let displacement = Int(expert).multipliedReportingOverflow(by: perExpert)
                        let offset = pair.packed.fileOffset.addingReportingOverflow(displacement.partialValue)
                        guard !displacement.overflow, !offset.overflow,
                            displacement.partialValue <= pair.packed.byteCount - perExpert
                        else {
                            throw JANGHFormatContract.ValidationError.invalid("JANGH expert offset overflow")
                        }
                        entry = try cached(Key(module: module, expert: expert)) {
                            if let views = diagnosticExpertViews[module] {
                                return Entry(array: views[Int(expert)], mappedBytes: perExpert)
                            }
                            return try map(fd: packedFD, offset: offset.partialValue, bytes: perExpert,
                                    shape: [1, packedHeader.shape[1], packedHeader.shape[2]], dtype: .uint32)
                        }
                        unique[expert] = entry
                        let sum = mappedBytes.addingReportingOverflow(entry.mappedBytes)
                        guard !sum.overflow else {
                            throw JANGHFormatContract.ValidationError.invalid("JANGH selected byte count overflow")
                        }
                        mappedBytes = sum.partialValue
                    }
                    packed.append(entry.array)
                }
                return Selection(ownerIdentity: identity, module: module, expertIDs: expertIDs, packed: packed,
                                 scales: scales.array, mappedPackedBytes: mappedBytes)
            }
        }
    }

    private func cached(_ key: Key, create: () throws -> Entry) throws -> Entry {
        if let found = entries[key] {
            cacheHits += 1
            leastRecentFirst.removeAll { $0 == key }
            leastRecentFirst.append(key)
            return found
        }
        cacheMisses += 1
        let value = try create()
        // Oversized views remain valid for this selection but are not cached.
        guard value.mappedBytes <= cacheByteLimit else { return value }
        while retainedBytes > cacheByteLimit - value.mappedBytes,
              let oldest = leastRecentFirst.first {
            leastRecentFirst.removeFirst()
            if let evicted = entries.removeValue(forKey: oldest) {
                retainedBytes -= evicted.mappedBytes
            }
        }
        entries[key] = value
        leastRecentFirst.append(key)
        retainedBytes += value.mappedBytes
        return value
    }

    private func map(fd: Int32, offset: Int, bytes: Int, shape: [Int], dtype: DType) throws -> Entry {
        let page = Int(getpagesize())
        let span = bytes.addingReportingOverflow(offset % page)
        guard offset >= 0, bytes > 0, !span.overflow, span.partialValue <= Int(Int32.max),
            shape.allSatisfy({ $0 > 0 && $0 <= Int(Int32.max) })
        else {
            throw JANGHFormatContract.ValidationError.invalid("invalid selected JANGH mapped span")
        }
        var dimensions = shape.map(Int32.init)
        var raw = mlx_array_new()
        do {
            let status = try withError {
                "/dev/fd/\(fd)".withCString { path in
                    mlx_array_new_mmap_file_region(&raw, path, UInt64(offset), bytes,
                                                  &dimensions, Int32(dimensions.count), dtype.cmlxDtype)
                }
            }
            guard status == 0 else {
                throw JANGHFormatContract.ValidationError.invalid("selected JANGH mapping failed")
            }
        } catch {
            mlx_array_free(raw)
            throw error
        }
        let array = MLXArray(raw)
        try JANGHBankLayout.requireReadyRowContiguous(array, role: "selected JANGH expert")
        return Entry(array: array, mappedBytes: span.partialValue)
    }
}
