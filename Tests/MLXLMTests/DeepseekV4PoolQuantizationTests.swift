// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@Suite("DSV4 segmented pool quantization", .serialized)
struct DeepseekV4PoolQuantizationTests {
    private static let hotLimit = 2 * 1024 * 1024

    @Test("production default is enabled with an explicit diagnostic opt-out")
    func defaultPolicy() {
        #expect(DeepseekV4Cache.resolvePoolQuantizationDefault(environment: [:]))
        #expect(DeepseekV4Cache.resolvePoolQuantizationDefault(
            environment: ["DSV4_POOL_QUANT": "1"]))
        #expect(!DeepseekV4Cache.resolvePoolQuantizationDefault(
            environment: ["DSV4_POOL_QUANT": "0"]))
        #expect(!DeepseekV4Cache.resolvePoolQuantizationDefault(
            environment: ["DSV4_POOL_QUANT": "off"]))

        let cache = DeepseekV4Cache(slidingWindow: 128, compressRatio: 4)
        #expect(cache.hybridPoolQuantizationEnabled)
        let diagnostic = DeepseekV4Cache(
            slidingWindow: 128, compressRatio: 4, poolQuantizationEnabled: false)
        #expect(!diagnostic.hybridPoolQuantizationEnabled)
    }

    @Test("small pools stay hot; promotion is segmented, bounded, and high fidelity")
    func adaptivePromotionAndQuality() {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }

        let cache = DeepseekV4Cache(
            slidingWindow: 128, compressRatio: 4, poolQuantizationEnabled: true)
        let short = Self.signal(rows: 1024, features: 512)
        #expect(short.nbytes < Self.hotLimit)
        cache.setPooled(.compressor, value: short)
        #expect(cache.hybridPoolQuantizedSegments(branch: .compressor) == nil)
        #expect(cache.hybridPoolRetainedByteCount(branch: .compressor) == short.nbytes)

        let raw = Self.signal(rows: 2049, features: 512)
        #expect(raw.nbytes > Self.hotLimit)
        cache.setPooled(.compressor, value: raw)
        guard let segments = cache.hybridPoolQuantizedSegments(branch: .compressor),
              let restored = cache.getPooled(.compressor)
        else {
            Issue.record("pool did not promote to encoded storage")
            return
        }
        MLX.eval(restored)
        #expect(segments.count == 33)
        #expect(segments.allSatisfy { $0.rowCount > 0 && $0.rowCount <= 64 })
        #expect(segments.allSatisfy { $0.bits == 8 && $0.groupSize == 32 })
        #expect(segments.allSatisfy { $0.codes.dtype == .uint8 })
        let retainedBefore = cache.hybridPoolRetainedByteCount(branch: .compressor)
        #expect(retainedBefore < Int(Double(raw.nbytes) * 0.60))
        #expect(Self.cosine(raw, restored) >= 0.999)

        // A materialized attention read must not be retained beside the codes.
        _ = cache.getPooled(.compressor)
        #expect(cache.hybridPoolRetainedByteCount(branch: .compressor) == retainedBefore)
        #expect(CacheStoreBudget.cacheBytes([cache as any KVCache])
            == cache.retainedCacheByteCount)
    }

    @Test("quantized trim keeps the prefix encoded")
    func trimWithoutTierChange() {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }

        let raw = Self.signal(rows: 2075, features: 512)
        let cache = DeepseekV4Cache(
            slidingWindow: 128, compressRatio: 4, poolQuantizationEnabled: true)
        cache.setPooled(.compressor, value: raw)
        #expect(cache.hybridPoolQuantizedSegments(branch: .compressor) != nil)
        _ = cache.trim(68) // max(1, 68 / 4) == 17 pool rows
        guard let kept = cache.getPooled(.compressor),
              let segments = cache.hybridPoolQuantizedSegments(branch: .compressor)
        else {
            Issue.record("trim unexpectedly expanded or cleared encoded pool")
            return
        }
        MLX.eval(kept)
        #expect(kept.shape == [1, 2058, 512])
        #expect(!segments.isEmpty)
        #expect(Self.cosine(raw[0..., 0..<2058, 0...], kept) >= 0.999)
    }

    @Test("SSD serializer preserves encoded segments instead of BF16 expansion")
    func encodedDiskRoundTrip() {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }

        let raw = Self.signal(rows: 2049, features: 512)
        let source = DeepseekV4Cache(
            slidingWindow: 128, compressRatio: 4, poolQuantizationEnabled: true)
        Self.fillRotating(source.local)
        source.setPooled(.compressor, value: raw)
        let sourceRetained = source.hybridPoolRetainedByteCount(branch: .compressor)

        let encoded = TQDiskSerializer.serialize(cache: [source])
        #expect(encoded["__dsv4_0_pool_comp_qcount__"] != nil)
        #expect(encoded["dsv4_0_pool_comp_q_0_codes"] != nil)
        #expect(encoded["dsv4_0_pool_comp_q_0_scales"] != nil)
        #expect(encoded["dsv4_0_pool_comp_q_0_biases"] != nil)

        let target = DeepseekV4Cache(
            slidingWindow: 128, compressRatio: 4, poolQuantizationEnabled: true)
        var targetLayers: [any KVCache] = [target]
        let restoredTokens = restoreFromDiskArrays(encoded, into: &targetLayers)
        #expect(restoredTokens == source.offset)
        guard let segments = target.hybridPoolQuantizedSegments(branch: .compressor),
              let restored = target.getPooled(.compressor)
        else {
            Issue.record("disk restore did not preserve encoded pool segments")
            return
        }
        MLX.eval(restored)
        #expect(segments.count == 33)
        #expect(target.hybridPoolRetainedByteCount(branch: .compressor) == sourceRetained)
        #expect(Self.cosine(raw, restored) >= 0.999)
    }

    @Test("one-million-token DSV4-Flash retained cache projection is below 10 GiB")
    func oneMillionTokenProjection() {
        let ratios = [0, 0] + (0..<41).map { $0.isMultiple(of: 2) ? 4 : 128 }
        let bytes = DeepseekV4Cache.projectedQuantizedCacheUpperBoundBytes(
            contextLength: 1_048_576,
            compressRatios: ratios,
            headDim: 512,
            indexerHeadDim: 128,
            slidingWindow: 128)
        #expect(ratios.count == 43)
        #expect(ratios.filter { $0 == 4 }.count == 21)
        #expect(ratios.filter { $0 == 128 }.count == 20)
        #expect(bytes > 3 * 1024 * 1024 * 1024)
        #expect(bytes < 10 * 1024 * 1024 * 1024)
    }

    private static func signal(rows: Int, features: Int) -> MLXArray {
        let count = rows * features
        return MLX.sin(MLXArray(0..<count).asType(.float32) * Float(0.013))
            .reshaped(1, rows, features)
            .asType(.bfloat16)
    }

    private static func cosine(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let a = lhs.asType(.float32).reshaped(-1)
        let b = rhs.asType(.float32).reshaped(-1)
        let numerator = (a * b).sum()
        let aa = (a * a).sum()
        let bb = (b * b).sum()
        MLX.eval([numerator, aa, bb])
        return numerator.item(Float.self)
            / max(sqrt(aa.item(Float.self) * bb.item(Float.self)), 1e-9)
    }

    private static func fillRotating(_ rotating: RotatingKVCache) {
        let keys = MLXArray.ones([1, 1, 5, 8], dtype: .bfloat16)
        let values = MLXArray.ones([1, 1, 5, 8], dtype: .bfloat16) * Float(2)
        _ = rotating.update(keys: keys, values: values)
    }
}
