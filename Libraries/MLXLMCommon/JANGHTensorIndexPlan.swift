import Foundation

/// Header-only admission for custom banks. This does not read tensor payloads,
/// construct modules, select an architecture, or enable a model factory.
struct JANGHTensorIndexPlan: Sendable {
    struct Dimensions: Sendable {
        let experts: Int
        let input: Int
        let output: Int
    }

    struct TensorHeader: Sendable {
        let dtype: String
        let shape: [Int]
        /// Safetensors offsets relative to the start of the data section.
        let start: Int
        let end: Int
    }

    struct ShardHeader: Sendable {
        let fileBytes: Int
        let dataStart: Int
        let tensors: [String: TensorHeader]
    }

    struct Location: Sendable {
        let shard: String
        let tensor: String
        let fileOffset: Int
        let byteCount: Int
    }

    struct Banks: Sendable {
        let packed: Location
        let scales: Location
    }

    let projections: [String: Banks]

    init(
        contract: JANGHFormatContract, dimensions: [String: Dimensions],
        weightMap: [String: String], shards: [String: ShardHeader]
    ) throws {
        typealias Failure = JANGHFormatContract.ValidationError
        guard Set(dimensions.keys) == Set(contract.projections.keys) else {
            throw Failure.invalid("JANGH architecture projection coverage mismatch")
        }
        let expectedNames = Set(contract.projections.keys.flatMap {
            [$0 + ".tq2_packed", $0 + ".tq2_scales"]
        })
        func isCustom(_ name: String) -> Bool {
            name.hasSuffix(".tq2_packed") || name.hasSuffix(".tq2_scales")
        }
        guard Set(weightMap.keys.filter(isCustom)) == expectedNames else {
            throw Failure.invalid("JANGH index projection coverage mismatch")
        }
        // Summaries must include every indexed shard, including ordinary weights,
        // so an unindexed or duplicate custom tensor cannot escape this audit.
        guard Set(weightMap.values) == Set(shards.keys) else {
            throw Failure.invalid("JANGH shard inventory mismatch")
        }
        var found = Set<String>()
        for (file, shard) in shards {
            guard !file.isEmpty, file != ".", file != "..",
                !file.contains("/"), !file.contains("\\"), file.hasSuffix(".safetensors"),
                shard.dataStart >= 8, shard.fileBytes >= shard.dataStart
            else { throw Failure.invalid("invalid JANGH shard summary") }
            var ranges: [(Int, Int)] = []
            for (name, header) in shard.tensors {
                guard header.start >= 0, header.end >= header.start,
                    header.end <= shard.fileBytes - shard.dataStart
                else { throw Failure.invalid("invalid safetensors byte range") }
                if header.end > header.start { ranges.append((header.start, header.end)) }
                if isCustom(name) {
                    guard expectedNames.contains(name), weightMap[name] == file,
                        found.insert(name).inserted
                    else { throw Failure.invalid("duplicate or unowned JANGH tensor") }
                }
            }
            ranges.sort { $0.0 < $1.0 }
            for index in 1..<max(1, ranges.count) {
                guard ranges[index - 1].1 <= ranges[index].0 else {
                    throw Failure.invalid("overlapping safetensors byte ranges")
                }
            }
        }
        guard found == expectedNames else {
            throw Failure.invalid("missing indexed JANGH tensor")
        }

        func locate(_ name: String, bytesPerElement: Int) throws -> (TensorHeader, Location) {
            guard let file = weightMap[name], let shard = shards[file],
                let header = shard.tensors[name]
            else { throw Failure.invalid("missing JANGH bank") }
            var count = bytesPerElement
            for size in header.shape {
                let product = count.multipliedReportingOverflow(by: size)
                guard size > 0, !product.overflow else {
                    throw Failure.invalid("JANGH bank size overflow")
                }
                count = product.partialValue
            }
            // Range bounds above guarantee this sum cannot overflow.
            let offset = shard.dataStart + header.start
            guard header.end - header.start == count, offset.isMultiple(of: bytesPerElement) else {
                throw Failure.invalid("JANGH bank byte count or dtype alignment mismatch")
            }
            return (header, Location(shard: file, tensor: name, fileOffset: offset, byteCount: count))
        }
        var result: [String: Banks] = [:]
        for (module, d) in dimensions {
            for suffix in [".tq_packed", ".tq_norms", ".weight"] {
                let old = module + suffix
                guard weightMap[old] == nil, !shards.values.contains(where: { $0.tensors[old] != nil })
                else { throw Failure.invalid("conflicting JANGH and legacy weight representation") }
            }
            let (packed, p) = try locate(module + ".tq2_packed", bytesPerElement: 4)
            let (scales, s) = try locate(module + ".tq2_scales", bytesPerElement: 2)
            try contract.validateTensorHeaders(
                module: module, experts: d.experts, inputDimensions: d.input,
                outputDimensions: d.output, packedShape: packed.shape, packedDType: packed.dtype,
                scalesShape: scales.shape, scalesDType: scales.dtype)
            result[module] = Banks(packed: p, scales: s)
        }
        projections = result
    }
}
