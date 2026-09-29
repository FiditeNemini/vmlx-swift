import Foundation
import MLX
import XCTest
@testable import MLXLMCommon

final class JANGHSelectedExpertTests: XCTestCase {
    private let parent = "model.layers.0.mlp.switch_mlp"
    private let ids: [UInt32] = [0, 3, 5, 9, 15, 1, 3, 14]

    private struct Fixture {
        let directory: URL
        let source: JANGHMappedBanks.SourceLease
        let packedBytes: Int
    }

    private func fixture(
        bits: [Int] = [2, 3, 4], inputRotation: String = "none",
        downRotation: String = "none", experts: Int = 16, width: Int = 64, hidden: Int = 32
    ) throws -> Fixture {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var books: [String: Any] = [:], quant: [String: Any] = [:], headers: [String: Any] = [:]
        var map: [String: String] = [:]
        var dimensions: [String: JANGHTensorIndexPlan.Dimensions] = [:]
        var payload = Data(), packedBytes = 0
        for (roleIndex, role) in ["gate_proj", "up_proj", "down_proj"].enumerated() {
            let b = bits[roleIndex], input = roleIndex == 2 ? hidden : width
            let output = roleIndex == 2 ? width : hidden
            let module = parent + "." + role
            let alpha = 0.125, beta = 0.0001
            books[String(b)] = ["alpha": alpha, "beta": beta, "levels": (0..<(1 << b)).map { code in
                let u = Double(code) - Double((1 << b) - 1) / 2
                return Double(Float(u * (alpha + beta * u * u)))
            }]
            quant[module] = ["mode": "jangtq2", "bits": b,
                             "rotation": roleIndex == 2 ? downRotation : inputRotation]
            dimensions[module] = .init(experts: experts, input: input, output: output)
            var words = [UInt32](repeating: 0, count: experts * output * input * b / 32)
            for i in 0..<(experts * output * input) {
                let code = UInt32((i * 17 + i / input * 7 + roleIndex * 11) % (1 << b))
                for bit in 0..<b where (code >> bit) & 1 != 0 {
                    let offset = i * b + bit
                    words[offset / 32] |= 1 << (offset % 32)
                }
            }
            let scales = (0..<(experts * output)).map { Float16(Float($0 % 7 + 1) / 8) }
            let packed = words.withUnsafeBytes { Data($0) }
            if roleIndex == 0 { packedBytes = packed.count }
            for (suffix, dtype, shape, bytes) in [
                ("tq2_packed", "U32", [experts, output, input * b / 32], packed),
                ("tq2_scales", "F16", [experts, output], scales.withUnsafeBytes { Data($0) }),
            ] {
                let name = module + "." + suffix
                headers[name] = ["dtype": dtype, "shape": shape,
                                 "data_offsets": [payload.count, payload.count + bytes.count]]
                map[name] = "model.safetensors"
                payload.append(bytes)
            }
        }
        var header = try JSONSerialization.data(withJSONObject: headers, options: .sortedKeys)
        guard header.count <= 4088 else { throw NSError(domain: "fixture", code: 1) }
        header.append(Data(repeating: 0x20, count: 4088 - header.count))
        var length = UInt64(header.count).littleEndian
        var file = withUnsafeBytes(of: &length) { Data($0) }
        file.append(header); file.append(payload)
        try file.write(to: directory.appendingPathComponent("model.safetensors"))
        try JSONSerialization.data(withJSONObject: ["weight_map": map]).write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))
        let config: [String: Any] = ["jangtq": ["version": 2, "packing": "lsb-bitstream",
            "scale_dtype": "float16", "codebook_family": "odd-cubic", "rotation": inputRotation,
            "codebooks": books], "quantization": quant]
        let contract = try JANGHFormatContract(configuration: JSONSerialization.data(withJSONObject: config))
        let metadata = try JANGHHeaderAdapter.read(directory: directory, indexName: "model.safetensors.index.json")
        let source = try JANGHMappedBanks.SourceLease(directory: directory, metadata: metadata,
                                                     contract: contract, dimensions: dimensions)
        return Fixture(directory: directory, source: source, packedBytes: packedBytes)
    }

    private func kernel(_ owner: JANGHExpertMappedBanks) throws -> JANGHSelectedExpertDecode {
        try JANGHSelectedExpertDecode(owner: owner, gateModule: parent + ".gate_proj",
                                     upModule: parent + ".up_proj", downModule: parent + ".down_proj")
    }

    func testLazyOwnerMapsOnlySelectedExpertSpansAndBoundsItsOwnCache() throws {
        try MLXMetalTestLock.withLock {
            let f = try fixture(experts: 64, width: 512, hidden: 256)
            defer { try? FileManager.default.removeItem(at: f.directory) }
            let before = MLX.Memory.activeMemory
            let owner = try JANGHExpertMappedBanks(source: f.source, cacheByteLimit: 256 * 1024)
            XCTAssertEqual(MLX.Memory.activeMemory, before)
            XCTAssertEqual(owner.cachedOwnedMappedBytes, 0)
            let s = try owner.selection(module: parent + ".gate_proj", expertIDs: ids)
            XCTAssertEqual(s.expertIDs, ids)
            XCTAssertTrue(s.packed[1] === s.packed[6])
            XCTAssertLessThan(s.mappedPackedBytes, f.packedBytes / 2)
            XCTAssertLessThanOrEqual(owner.cachedOwnedMappedBytes, 256 * 1024)
            XCTAssertGreaterThan(MLX.Memory.activeMemory, before)
            XCTAssertThrowsError(try owner.selection(module: parent + ".gate_proj", expertIDs: [0]))
            XCTAssertThrowsError(try owner.selection(module: parent + ".gate_proj", expertIDs: Array(repeating: 64, count: 8)))
            XCTAssertThrowsError(try owner.selection(module: "unknown", expertIDs: ids))
            owner.removeAllCachedViews()
            XCTAssertEqual(owner.cachedOwnedMappedBytes, 0)
            XCTAssertEqual(s.packed[0].shape, [1, 256, 32])
        }
    }

    func testSelectedDecodeMatchesProvenWholeBankCompositionAcrossRotationsAndDTypes() throws {
        try MLXMetalTestLock.withLock {
            for bits in [[2, 2, 2], [2, 3, 4]] {
                for inputRotation in ["none", "hadamard32"] {
                    for downRotation in ["none", "hadamard32"] {
                        let f = try fixture(bits: bits, inputRotation: inputRotation, downRotation: downRotation)
                        defer { try? FileManager.default.removeItem(at: f.directory) }
                        let owner = try JANGHExpertMappedBanks(source: f.source, cacheByteLimit: 32768)
                        let selected = try kernel(owner)
                        let whole = try JANGHMappedBanks(source: f.source)
                        let reference = try JANGHRoutedDecodeBlock(banks: whole, parentModule: parent, activationLimit: 0.75)
                        let gate = try owner.selection(module: parent + ".gate_proj", expertIDs: ids)
                        let up = try owner.selection(module: parent + ".up_proj", expertIDs: ids)
                        let down = try owner.selection(module: parent + ".down_proj", expertIDs: ids)
                        let routes = MLXArray(ids, [1, 8])
                        let scores = MLXArray([Float(0.05), 0.10, 0.15, 0.20, 0.10, 0.05, 0.15, 0.20], [1, 8])
                        for inputDType in [DType.float16, .bfloat16, .float32] {
                            let x = MLXArray((0..<64).map { Float(($0 * 19) % 59 - 29) / 31 }, [1, 64]).asType(inputDType)
                            let h = try selected.activatePreparedInput(selected.prepareInput(x), gate: gate, up: up, limit: 0.75)
                            for outputDType in [DType.float16, .bfloat16, .float32] {
                                let actual = try selected.projectPreparedHidden(h, down: down, scores: scores, outputDType: outputDType)
                                let expected = try reference(x, indices: routes, scores: scores, outputDType: outputDType)
                                let a = actual.asType(.float32).asArray(Float.self)
                                let e = expected.asType(.float32).asArray(Float.self)
                                XCTAssertEqual(a.count, 64)
                                let tolerance: Float = outputDType == .bfloat16 ? 0.008 : (outputDType == .float16 ? 0.002 : 0.0005)
                                for i in a.indices {
                                    XCTAssertTrue(a[i].isFinite)
                                    XCTAssertEqual(a[i], e[i], accuracy: max(0.00005, abs(e[i]) * tolerance),
                                                   "bits=\(bits) rotations=\(inputRotation)/\(downRotation) input=\(inputDType) output=\(outputDType) row=\(i)")
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    func testSelectionsCannotCrossModelOwnersEvenWithIdenticalMetadata() throws {
        try MLXMetalTestLock.withLock {
            let f = try fixture()
            defer { try? FileManager.default.removeItem(at: f.directory) }
            let a = try JANGHExpertMappedBanks(source: f.source, cacheByteLimit: 0)
            let b = try JANGHExpertMappedBanks(source: f.source, cacheByteLimit: 0)
            let ka = try kernel(a), kb = try kernel(b)
            let x = MLXArray.ones([1, 64])
            let ga = try a.selection(module: parent + ".gate_proj", expertIDs: ids)
            let ua = try a.selection(module: parent + ".up_proj", expertIDs: ids)
            let ub = try b.selection(module: parent + ".up_proj", expertIDs: ids)
            XCTAssertThrowsError(try ka.activatePreparedInput(x, gate: ga, up: ub, limit: nil))
            let h = try ka.activatePreparedInput(x, gate: ga, up: ua, limit: nil)
            let db = try b.selection(module: parent + ".down_proj", expertIDs: ids)
            XCTAssertThrowsError(try kb.projectPreparedHidden(h, down: db, scores: MLXArray.ones([1, 8]), outputDType: .float32))
        }
    }

    func testCachedViewsRevalidateTheSourceBeforeReuse() throws {
        try MLXMetalTestLock.withLock {
            let f = try fixture()
            defer { try? FileManager.default.removeItem(at: f.directory) }
            let owner = try JANGHExpertMappedBanks(source: f.source, cacheByteLimit: 1 << 20)
            let held = try owner.selection(module: parent + ".gate_proj", expertIDs: ids)
            let path = f.directory.appendingPathComponent("model.safetensors")
            let bytes = try Data(contentsOf: path)
            try bytes.write(to: path, options: .atomic)
            XCTAssertThrowsError(try owner.selection(module: parent + ".gate_proj", expertIDs: ids))
            // Existing allocations retain the original inode and remain readable.
            XCTAssertEqual(held.packed[0].asArray(UInt32.self).count, 128)
        }
    }

    func testZeroCacheSelectionsSurviveOwnerReleaseAndUnlink() throws {
        try MLXMetalTestLock.withLock {
            let f = try fixture()
            defer { try? FileManager.default.removeItem(at: f.directory) }
            var owner: JANGHExpertMappedBanks? = try JANGHExpertMappedBanks(source: f.source, cacheByteLimit: 0)
            let selection = try owner!.selection(module: parent + ".gate_proj", expertIDs: ids)
            XCTAssertEqual(owner!.cachedOwnedMappedBytes, 0)
            let expected = selection.packed[0].asArray(UInt32.self)
            owner = nil
            try FileManager.default.removeItem(at: f.directory.appendingPathComponent("model.safetensors"))
            XCTAssertEqual(selection.packed[0].asArray(UInt32.self), expected)
        }
    }
}
