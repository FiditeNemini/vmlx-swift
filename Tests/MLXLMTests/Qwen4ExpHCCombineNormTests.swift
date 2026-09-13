// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

@Suite("Flash AR HC residual and next normalization", .serialized)
struct Qwen4ExpHCCombineNormTests {
    private func bits(_ value: MLXArray) -> [UInt32] {
        value.asType(.float32).asArray(Float.self).map(\.bitPattern)
    }

    @Test("both outputs preserve MLX rounding, group reduction, and source ownership")
    func exactArithmetic() throws {
        try MLXMetalTestLock.withLock {
            var checks = 0
            for dtype in [DType.bfloat16, .float16] {
                for streams in [1, 2, 4] {
                    for width in [32, 64, 96, 128, 192, 768, 2560, 4096] {
                        let kernel = Qwen4ExpHCCombineNorm(hiddenSize: width, eps: 1e-6)
                        for stride in [1, 2] {
                            for magnitude: Float in [0, 0.0001, 1, 8] {
                                for blockType in [dtype, .float32] {
                                    let coords = MLXArray(0 ..< (streams * width * stride)).asType(
                                        .float32)
                                    let residual = (sin(coords * 0.173 + 0.27) * magnitude)
                                        .asType(dtype).reshaped(1, 1, -1)[
                                            .ellipsis, .stride(by: stride)]
                                    let block =
                                        (cos(coords[0 ..< (width * stride)] * 0.071) * magnitude)
                                        .asType(blockType).reshaped(1, 1, -1)[
                                            .ellipsis, .stride(by: stride)]
                                    let injection =
                                        (sin(coords[0 ..< (streams * stride)] * 0.37) + 1)
                                        .asType(dtype).reshaped(1, 1, -1)[
                                            .ellipsis, .stride(by: stride)]
                                    let weight = (cos(coords * 0.097) + 1.3)
                                        .asType(dtype)[.stride(by: stride)]
                                    let original = [residual, block, injection, weight].map(bits)
                                    let product =
                                        expandedDimensions(block, axis: -2)
                                        * expandedDimensions(injection, axis: -1)
                                    let expected = (residual + product.reshaped(residual.shape))
                                        .asType(dtype)
                                    let expectedNorm =
                                        MLXFast.rmsNorm(
                                            expected.reshaped(1, 1, streams, width),
                                            weight: .mlxNone, eps: 1e-6
                                        )
                                        .reshaped(expected.shape) * weight
                                    let actual = try #require(
                                        kernel(
                                            residual: residual, block: block, injection: injection,
                                            weight: weight))
                                    MLX.eval(
                                        expected, expectedNorm, actual.residual, actual.normalized)
                                    let residualMatches = bits(actual.residual) == bits(expected)
                                    let normMatches = bits(actual.normalized) == bits(expectedNorm)
                                    #expect(
                                        residualMatches,
                                        "dtype=\(dtype) block=\(blockType) streams=\(streams) width=\(width) stride=\(stride) magnitude=\(magnitude)"
                                    )
                                    #expect(
                                        normMatches,
                                        "dtype=\(dtype) block=\(blockType) streams=\(streams) width=\(width) stride=\(stride) magnitude=\(magnitude)"
                                    )
                                    #expect(actual.residual.shape == residual.shape)
                                    #expect(actual.normalized.dtype == dtype)
                                    #expect(
                                        [residual, block, injection, weight].map(bits) == original)
                                    checks += 1
                                }
                            }
                        }
                    }
                }
            }
            #expect(checks == 768)
            print("[HCCombineNorm] exact_cases=\(checks) tokens_per_second=NA reason=no_generation")
        }
    }

    @Test("unsupported widths, dtypes, prefill, batch and tracing fall back")
    func fallback() throws {
        try MLXMetalTestLock.withLock {
            for (batch, rows, width, dtype) in [
                (1, 2, 128, DType.bfloat16), (2, 1, 128, .bfloat16),
                (1, 1, 31, .bfloat16), (1, 1, 8192, .bfloat16), (1, 1, 128, .float32),
            ] {
                let kernel = Qwen4ExpHCCombineNorm(hiddenSize: width, eps: 1e-6)
                #expect(
                    kernel(
                        residual: MLXArray.zeros([batch, rows, 4 * width], dtype: dtype),
                        block: MLXArray.zeros([batch, rows, width], dtype: dtype),
                        injection: MLXArray.zeros([batch, rows, 4], dtype: dtype),
                        weight: MLXArray.ones([4 * width], dtype: dtype)) == nil)
            }
            let kernel = Qwen4ExpHCCombineNorm(hiddenSize: 128, eps: 1e-6)
            let residual = MLXArray.zeros([1, 1, 512], dtype: .bfloat16)
            let block = MLXArray.zeros([1, 1, 128], dtype: .bfloat16)
            let injection = MLXArray.ones([1, 1, 4], dtype: .bfloat16)
            let weight = MLXArray.ones([512], dtype: .bfloat16)
            #expect(
                kernel(
                    residual: residual, block: block, injection: injection,
                    weight: weight.asType(.float16)) == nil)
            CompiledDecodeTrace.withActive {
                #expect(
                    kernel(residual: residual, block: block, injection: injection, weight: weight)
                        == nil)
            }
            Device.withDefaultDevice(.cpu) {
                #expect(
                    kernel(residual: residual, block: block, injection: injection, weight: weight)
                        == nil)
            }
        }
    }
}
