// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite("native affine active-expert SSD routing", .serialized)
struct AffineStreamingSwitchGLUTests {
    private final class ConcurrentResults: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var shapes: [[Int]] = []
        private(set) var failures: [String] = []

        func append(shape: [Int]) {
            lock.lock()
            shapes.append(shape)
            lock.unlock()
        }

        func append(failure: String) {
            lock.lock()
            failures.append(failure)
            lock.unlock()
        }
    }

    private struct Fixture {
        let directory: URL
        let tensors: [String: MLXArray]
    }

    private func makeFixture(numExperts: Int = 4, dimensions: Int = 32) throws -> Fixture {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("affine-streaming-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        func dense(expert: Int, projection: Int) -> MLXArray {
            let count = dimensions * dimensions
            let values = (0 ..< count).map { index -> Float in
                let centered = Float((index * 13 + expert * 17 + projection * 23) % 101) - 50
                return centered / Float(200 + projection * 40)
            }
            return MLXArray(values, [dimensions, dimensions])
        }

        var tensors: [String: MLXArray] = [:]
        for (projectionIndex, projection) in ["gate_proj", "up_proj", "down_proj"].enumerated() {
            var weights: [MLXArray] = []
            var scales: [MLXArray] = []
            var biases: [MLXArray] = []
            for expert in 0 ..< numExperts {
                let quantized = MLX.quantized(
                    dense(expert: expert, projection: projectionIndex),
                    groupSize: dimensions, bits: 4, mode: .affine)
                weights.append(quantized.wq)
                scales.append(quantized.scales.asType(.float16))
                biases.append(quantized.biases!.asType(.float16))
            }
            let base = "language_model.layers.0.mlp.switch_mlp.\(projection)"
            tensors[base + ".weight"] = stacked(weights, axis: 0)
            tensors[base + ".scales"] = stacked(scales, axis: 0)
            tensors[base + ".biases"] = stacked(biases, axis: 0)
        }
        MLX.eval(Array(tensors.values))

        let shardName = "model.safetensors"
        let shardURL = directory.appendingPathComponent(shardName)
        try MLX.save(arrays: tensors, url: shardURL)
        // Production Qwen4 uses JangPressPrestacker's alignment overlay.
        // Reproduce that contract here by padding the safetensors JSON header
        // so the data segment begins on a page boundary without changing any
        // tensor-relative data_offsets.
        let saved = try Data(contentsOf: shardURL)
        let oldHeaderLength = saved.prefix(8).withUnsafeBytes {
            $0.loadUnaligned(as: UInt64.self).littleEndian
        }
        var header = saved.subdata(in: 8 ..< (8 + Int(oldHeaderLength)))
        let padding = (4096 - ((8 + header.count) % 4096)) % 4096
        header.append(Data(repeating: 0x20, count: padding))
        var aligned = Data()
        var newHeaderLength = UInt64(header.count).littleEndian
        aligned.append(contentsOf: withUnsafeBytes(of: &newHeaderLength) { Array($0) })
        aligned.append(header)
        aligned.append(saved.suffix(from: 8 + Int(oldHeaderLength)))
        try aligned.write(to: shardURL)
        let weightMap = Dictionary(uniqueKeysWithValues: tensors.keys.map { ($0, shardName) })
        let index = ["weight_map": weightMap]
        try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        return Fixture(directory: directory, tensors: tensors)
    }

    @Test("loader exclusion is exact and does not swallow shared experts")
    func exactLoaderExclusion() {
        #expect(AffineStreamingExpertCatalog.isRoutedTensorKey(
            "language_model.layers.4.mlp.switch_mlp.gate_proj.weight"))
        #expect(AffineStreamingExpertCatalog.isRoutedTensorKey(
            "model.language_model.layers.4.mlp.switch_mlp.down_proj.biases"))
        #expect(!AffineStreamingExpertCatalog.isRoutedTensorKey(
            "language_model.layers.4.mlp.shared_expert.gate_proj.weight"))
        #expect(!AffineStreamingExpertCatalog.isRoutedTensorKey(
            "language_model.layers.4.mlp.switch_mlp.gate_proj.tq_packed"))
    }

    @Test("exact expert regions preserve projection geometry")
    func exactRegionGeometry() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let catalog = AffineStreamingExpertCatalog(
            numExperts: 4, inputDims: 32, hiddenDims: 32)
        try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
        let expert = try catalog.loadExpert(layer: 0, expert: 3)
        #expect(expert.gate.weight.shape == [32, 4])
        #expect(expert.gate.scales.shape == [32, 1])
        #expect(expert.gate.biases.shape == [32, 1])
        #expect(expert.gate.bits == 4)
        #expect(expert.gate.groupSize == 32)
    }

    @Test("selected SSD experts match the full native affine bank")
    func numericalParityAndBoundedCache() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let catalog = AffineStreamingExpertCatalog(
            numExperts: 4, inputDims: 32, hiddenDims: 32)
        try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
        let streaming = AffineStreamingSwitchGLU(
            inputDims: 32, hiddenDims: 32, numExperts: 4, layerIndex: 0,
            catalog: catalog, cacheExpertLimit: 2, prefillChunkSize: 1)

        let input = (MLXArray(0 ..< 32).asType(.float32) / 64).reshaped([1, 1, 32])
        let indices = MLXArray([Int32(3), Int32(1)], [1, 1, 2]).asType(.uint32)
        let scores = MLXArray([Float(0.65), Float(0.35)], [1, 1, 2])
        let actual = streaming.reduced(input, indices: indices, scores: scores)

        func tensor(_ projection: String, _ suffix: String) -> MLXArray {
            fixture.tensors[
                "language_model.layers.0.mlp.switch_mlp.\(projection).\(suffix)"]!
        }
        let expanded = expandedDimensions(input, axes: [-2, -3])
        func project(_ projection: String, _ value: MLXArray) -> MLXArray {
            MLX.gatherQuantizedMM(
                value, tensor(projection, "weight"),
                scales: tensor(projection, "scales"),
                biases: tensor(projection, "biases"), rhsIndices: indices,
                transpose: true, groupSize: 32, bits: 4, mode: .affine)
        }
        let activated = silu(project("gate_proj", expanded))
            * project("up_proj", expanded)
        let routed = squeezed(project("down_proj", activated), axis: -2)
        let expected = (routed * scores[.ellipsis, .newAxis]).sum(axis: -2)
        MLX.eval(expected)

        #expect(MLX.allClose(actual, expected, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        #expect(streaming.cachedExpertCount == 2)

        _ = streaming.reduced(
            input,
            indices: MLXArray([Int32(0), Int32(2)], [1, 1, 2]).asType(.uint32),
            scores: scores)
        #expect(streaming.cachedExpertCount == 2)
    }

    @Test("catalog supports concurrent exact-region opens")
    func concurrentRegionAccess() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let catalog = AffineStreamingExpertCatalog(
            numExperts: 4, inputDims: 32, hiddenDims: 32)
        try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
        let results = ConcurrentResults()
        DispatchQueue.concurrentPerform(iterations: 16) { iteration in
            do {
                let arrays = try catalog.loadExpert(layer: 0, expert: iteration % 4)
                results.append(shape: arrays.down.weight.shape)
            } catch {
                results.append(failure: String(describing: error))
            }
        }
        #expect(results.failures.isEmpty)
        #expect(results.shapes.count == 16)
        #expect(results.shapes.allSatisfy { $0 == [32, 4] })
    }

    @Test("optional real Qwen4 overlay validates every affine expert descriptor")
    func optionalRealBundleGeometry() throws {
        guard let path = ProcessInfo.processInfo.environment["QWEN4_REAL_MODEL"],
            !path.isEmpty
        else { return }
        let catalog = AffineStreamingExpertCatalog(
            numExperts: 512, inputDims: 2560, hiddenDims: 640)
        try catalog.configure(
            modelDirectory: URL(fileURLWithPath: path, isDirectory: true),
            layerCount: 48)
        let first = try catalog.loadExpert(layer: 0, expert: 0)
        let last = try catalog.loadExpert(layer: 47, expert: 511)
        #expect(first.gate.weight.shape == [640, 320])
        #expect(first.down.weight.shape == [2560, 80])
        #expect(last.gate.groupSize == 64)
        #expect(last.gate.bits == 4)
        #expect(last.down.groupSize == 64)
        #expect(last.down.bits == 4)
    }
}
