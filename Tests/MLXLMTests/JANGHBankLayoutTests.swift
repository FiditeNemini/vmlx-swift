import Cmlx
import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class JANGHBankLayoutTests: XCTestCase {
    private let prefix = "model.layers.0.mlp.switch_mlp"

    private func contract() throws -> JANGHFormatContract {
        let book: [String: Any] = ["alpha": 1.0, "beta": 0.0,
                                   "levels": (0 ..< 8).map { Double($0) - 3.5 }]
        let root: [String: Any] = [
            "jangtq": ["version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
                       "codebook_family": "odd-cubic", "rotation": "none", "codebooks": ["3": book]],
            "quantization": Dictionary(uniqueKeysWithValues: ["gate_proj", "up_proj", "down_proj"].map {
                (prefix + "." + $0, ["mode": "jangtq2", "bits": 3, "rotation": "none"] as [String: Any])
            }),
        ]
        return try JANGHFormatContract(configuration: JSONSerialization.data(withJSONObject: root))
    }

    private func available(_ array: MLXArray) -> Bool {
        var value = false
        XCTAssertEqual(_mlx_array_is_available(&value, array.ctx), 0)
        return value
    }

    private func rowContiguous(_ array: MLXArray) -> Bool {
        var value = false
        XCTAssertEqual(_mlx_array_is_row_contiguous(&value, array.ctx), 0)
        return value
    }

    /// Four slots permit independently checking gate/up packed and scale banks.
    private func calls(_ banks: [MLXArray]) throws -> [() throws -> MLXArray] {
        let c = try contract()
        let single = try JANGHProjectionKernel(contract: c, module: prefix + ".gate_proj")
        let fused = try JANGHFusedGateUpKernel(contract: c, gateModule: prefix + ".gate_proj",
                                              upModule: prefix + ".up_proj", outputRotation: .none)
        let down = try JANGHWeightedDownKernel(contract: c, module: prefix + ".down_proj")
        let x = MLXArray([Float](repeating: 1, count: 32), [1, 32])
        let hidden = MLXArray([Float](repeating: 1, count: 64), [2, 32])
        let indices = MLXArray([UInt32(1), 0], [1, 2])
        let scores = MLXArray([Float(0.5), 0.5], [1, 2])
        return [
            { try single.project(x, packed: banks[0], scales: banks[1], indices: indices) },
            { try fused.activatePreparedInput(x, gatePacked: banks[0], gateScales: banks[1],
                                               upPacked: banks[2], upScales: banks[3],
                                               indices: indices, limit: nil) },
            { try down.projectPreparedHidden(hidden, preparedBasis: .none, packed: banks[0],
                                               scales: banks[1], indices: indices, scores: scores,
                                               outputDType: .float32) },
        ]
    }

    private func readyBanks() -> [MLXArray] {
        let packed = MLXArray([UInt32](repeating: 0, count: 54), [2, 9, 3])
        let scales = MLXArray([Float16](repeating: 1, count: 18), [2, 9])
        return [packed, scales, packed, scales]
    }

    private func expectRejected(_ body: () throws -> MLXArray, containing reason: String) {
        XCTAssertThrowsError(try body()) { error in
            guard case JANGHFormatContract.ValidationError.invalid(let message) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertTrue(message.contains(reason), message)
        }
    }

    func testReadyDenseAndOffsetBanksRemainAccepted() throws {
        try MLXMetalTestLock.withLock {
            let dense = readyBanks()
            for bank in dense {
                XCTAssertTrue(available(bank))
                XCTAssertTrue(rowContiguous(bank))
            }
            let denseOutputs = try calls(dense).map { try $0().asArray(Float.self) }
            XCTAssertTrue(denseOutputs[0].allSatisfy { $0 == -112 })
            XCTAssertTrue(denseOutputs[1].allSatisfy(\.isFinite))
            XCTAssertTrue(denseOutputs[2].allSatisfy { $0 == -112 })

            let packedBase = MLXArray([UInt32](repeating: 0, count: 81), [3, 9, 3])
            let scaleBase = MLXArray([Float16](repeating: 1, count: 27), [3, 9])
            let packedView = packedBase[1 ..< 3, 0..., 0...]
            let scaleView = scaleBase[1 ..< 3, 0...]
            // Only this tiny test view is explicitly made ready; the primitive never evaluates banks.
            eval(packedView, scaleView)
            XCTAssertTrue(rowContiguous(packedView))
            XCTAssertTrue(rowContiguous(scaleView))
            let offsetOutputs = try calls([packedView, scaleView, packedView, scaleView]).map {
                try $0().asArray(Float.self)
            }
            XCTAssertEqual(offsetOutputs, denseOutputs)
        }
    }

    func testSameShapeNoncontiguousPackedAndScaleBanksRefuse() throws {
        try MLXMetalTestLock.withLock {
            let packed = MLXArray([UInt32](repeating: 0, count: 54), [2, 3, 9]).transposed(0, 2, 1)
            let scales = MLXArray([Float16](repeating: 1, count: 18), [9, 2]).transposed()
            eval(packed, scales)
            XCTAssertEqual(packed.shape, [2, 9, 3])
            XCTAssertEqual(scales.shape, [2, 9])
            XCTAssertTrue(available(packed))
            XCTAssertTrue(available(scales))
            XCTAssertFalse(rowContiguous(packed))
            XCTAssertFalse(rowContiguous(scales))
            for slot in 0 ..< 4 {
                var banks = readyBanks()
                banks[slot] = slot.isMultiple(of: 2) ? packed : scales
                let candidates = try calls(banks)
                for index in slot < 2 ? [0, 1, 2] : [1] {
                    expectRejected(candidates[index], containing: "not row contiguous")
                }
            }
        }
    }

    func testUnavailableBanksAreNotEvaluatedOnRejection() throws {
        try MLXMetalTestLock.withLock {
            for slot in 0 ..< 4 {
                var banks = readyBanks()
                let lazy = slot.isMultiple(of: 2)
                    ? MLXArray.zeros([2, 9, 3], dtype: .uint32)
                    : MLXArray.ones([2, 9], dtype: .float16)
                XCTAssertFalse(available(lazy))
                banks[slot] = lazy
                let candidates = try calls(banks)
                for index in slot < 2 ? [0, 1, 2] : [1] {
                    expectRejected(candidates[index], containing: "unavailable")
                    XCTAssertFalse(available(lazy), "Metadata guard must not evaluate the bank")
                }
            }
        }
    }
}
