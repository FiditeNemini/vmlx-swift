import Foundation
import MLX
import MLXLMCommon

/// Model-owned transactional pair: sparse indexer rows cannot be restored or
/// trimmed independently of attention KV. Sliding caches retain window-1 rows.
/// Batch conversion and speculative rollback after window wrap are not enabled.
final class NaiveN05FlashCache: DiskCacheStateProviding {
    enum Failure: Error { case invalidGeometry, invalidCompanion }
    private(set) var offset: Int = 0
    var maxSize: Int? { window }
    func innerState() -> [MLXArray] { rows }
    let window: Int?
    let requiresIndexer: Bool
    private var rows: [MLXArray] = []
    var keyOffset: Int { offset - (rows.first?.dim(2) ?? 0) }
    init(window: Int?, requiresIndexer: Bool) {
        precondition(window == nil || window! > 0)
        self.window = window
        self.requiresIndexer = requiresIndexer
    }
    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let length = (rows.first?.dim(2) ?? 0) + n
        let padding = MLXArray.ones([1, offset + n], dtype: .bool)
        return .array(NaiveN05FlashMath.allowedMask(padding: padding, queryOffset: offset,
            length: n, keyOffset: keyOffset, keyLength: length, window: window ?? windowSize))
    }
    var diskCacheStateIdentifier: String { "naive-n05-paired-v1-window-\(window ?? 0)-indexer-\(requiresIndexer)" }
    var state: [MLXArray] {
        get { rows }
        set {
            // Generic restore must use the validated DiskCacheStateProviding API.
            precondition(newValue.isEmpty || valid(newValue, offset: offset))
            rows = newValue
        }
    }
    var metaState: [String] {
        get { [diskCacheStateIdentifier, String(offset)] }
        set {
            precondition(newValue.count == 2 && newValue[0] == diskCacheStateIdentifier)
            offset = Int(newValue[1])!
        }
    }
    private func valid(_ arrays: [MLXArray], offset: Int) -> Bool {
        if offset == 0 && arrays.isEmpty { return true }
        guard offset >= 0, arrays.count == (requiresIndexer ? 3 : 2),
            arrays.allSatisfy({ $0.ndim == 4 }), let key = arrays.first,
            key.dim(2) <= offset, arrays[1].dim(0) == key.dim(0),
            arrays[1].dim(1) == key.dim(1), arrays[1].dim(2) == key.dim(2)
        else { return false }
        if let window, key.dim(2) != min(offset, window - 1) { return false }
        if window == nil && key.dim(2) != offset { return false }
        if requiresIndexer {
            let index = arrays[2]
            guard index.dim(0) == key.dim(0), index.dim(1) == 1,
                index.dim(2) == key.dim(2), index.dtype == .float32
            else { return false }
        }
        return true
    }
    func restoreDiskCacheState(_ state: [MLXArray], metadata: [String], offset: Int) -> Bool {
        guard metadata == [diskCacheStateIdentifier, String(offset)], valid(state, offset: offset) else { return false }
        // Validate the entire pair before touching either leaf.
        rows = state
        self.offset = offset
        return true
    }
    func append(keys: MLXArray, values: MLXArray, indexer: MLXArray?) throws -> (MLXArray, MLXArray, MLXArray?) {
        guard keys.ndim == 4, values.ndim == 4, keys.dim(2) > 0,
            keys.shape.dropLast() == values.shape.dropLast() else { throw Failure.invalidGeometry }
        guard requiresIndexer == (indexer != nil) else { throw Failure.invalidCompanion }
        if let indexer {
            guard indexer.ndim == 4, indexer.dim(0) == keys.dim(0), indexer.dim(1) == 1,
                indexer.dim(2) == keys.dim(2), indexer.dtype == .float32 else { throw Failure.invalidCompanion }
        }
        let incoming = [keys, values] + (indexer.map { [$0] } ?? [])
        if !rows.isEmpty {
            guard zip(rows, incoming).allSatisfy({ old, new in
                old.dim(0) == new.dim(0) && old.dim(1) == new.dim(1)
                && old.dim(3) == new.dim(3) && old.dtype == new.dtype
            }) else { throw Failure.invalidGeometry }
        }
        let full = rows.isEmpty ? incoming : zip(rows, incoming).map { concatenated([$0, $1], axis: 2) }
        let nextOffset = offset + keys.dim(2)
        let retained: [MLXArray]
        if let window {
            let start = max(0, full[0].dim(2) - (window - 1))
            retained = full.map { $0[0..., 0..., start..., 0...] }
        } else { retained = full }
        rows = retained
        offset = nextOffset
        return (full[0], full[1], requiresIndexer ? full[2] : nil)
    }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        precondition(!requiresIndexer, "Sparse Naive cache requires atomic companion append")
        let result = try! append(keys: keys, values: values, indexer: nil)
        return (result.0, result.1)
    }
    var isTrimmable: Bool { window == nil }
    func trim(_ n: Int) -> Int {
        guard n > 0, window == nil, !rows.isEmpty else { return 0 }
        let count = min(n, offset)
        rows = rows.map { $0[0..., 0..., ..<(offset - count), 0...] }
        offset -= count
        return count
    }
    func copy() -> any KVCache {
        let result = NaiveN05FlashCache(window: window, requiresIndexer: requiresIndexer)
        // No in-place writes in this reference cache; immutable snapshots own their graph.
        result.rows = rows.map { $0 + MLXArray(0, dtype: $0.dtype) }
        result.offset = offset
        return result
    }
}
