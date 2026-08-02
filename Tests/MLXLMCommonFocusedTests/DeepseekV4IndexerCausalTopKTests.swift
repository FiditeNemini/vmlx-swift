// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLX
@testable import MLXLLM
import MLXLMCommon
import Testing

@Suite("DSV4 indexer causal top-k", .serialized)
struct DeepseekV4IndexerCausalTopKTests {

    @Test("prompt-minus-one typed disk seed matches cold DSV4 chunking")
    func promptMinusOneDiskSeedMatchesColdPrefill() {
        FocusedMLXTestSupport.withLock {
            var cfg = DeepseekV4Configuration()
            cfg.hiddenSize = 8
            cfg.numAttentionHeads = 2
            cfg.headDim = 4
            cfg.qkRopeHeadDim = 2
            cfg.qLoraRank = 4
            cfg.oGroups = 2
            cfg.oLoraRank = 2
            cfg.indexNHeads = 2
            cfg.indexHeadDim = 4
            cfg.indexTopk = 2
            cfg.slidingWindow = 8
            cfg.compressRatios = [4]

            let attention = DeepseekV4Attention(config: cfg, layerIdx: 0)
            let values = (0..<(25 * cfg.hiddenSize)).map {
                Float(($0 % 29) - 14) / 32
            }
            let input = MLXArray(values).asType(.float16)
                .reshaped(1, 25, cfg.hiddenSize)

            func cache() -> DeepseekV4Cache {
                DeepseekV4Cache(
                    slidingWindow: cfg.slidingWindow,
                    compressRatio: 4,
                    poolQuantizationEnabled: false)
            }

            // Existing cold prefill shape for step=16: 16 tokens, then the
            // final 9-token remainder in one forward.
            let cold = cache()
            _ = attention(input[0..., ..<16, 0...], mask: .none, cache: cold)
            let coldTail = attention(
                input[0..., 16..., 0...], mask: .none, cache: cold)

            // New capture shape: the same first 16 tokens, then 8 tokens to
            // snapshot exact N-1 state, followed by the final token.
            let seed = cache()
            _ = attention(input[0..., ..<16, 0...], mask: .none, cache: seed)
            _ = attention(input[0..., 16..<24, 0...], mask: .none, cache: seed)
            MLX.eval(seed)
            let encoded = TQDiskSerializer.serialize(cache: [seed])

            var restored: [any KVCache] = [cache()]
            let restoredTokens = restoreFromDiskArrays(encoded, into: &restored)
            let restoredSeed = restored[0] as! DeepseekV4Cache
            let warmTail = attention(
                input[0..., 24..., 0...], mask: .none, cache: restoredSeed)
            MLX.eval(coldTail, warmTail)

            let coldLast = coldTail[0, -1, 0...]
            let warmLast = warmTail[0, -1, 0...]
            let maxError = (coldLast - warmLast).abs().max().item(Float.self)
            #expect(restoredTokens == 24)
            #expect(restoredSeed.offset == 25)
            #expect(maxError < 2e-3)
        }
    }

    @Test("ratio-4 attention preserves indexer history across the top-k boundary")
    func attentionAdvancesIndexerHistoryBeforeTopKSelection() {
        FocusedMLXTestSupport.withLock {
            var cfg = DeepseekV4Configuration()
            cfg.hiddenSize = 8
            cfg.numAttentionHeads = 2
            cfg.headDim = 4
            cfg.qkRopeHeadDim = 2
            cfg.qLoraRank = 4
            cfg.oGroups = 2
            cfg.oLoraRank = 2
            cfg.indexNHeads = 2
            cfg.indexHeadDim = 4
            cfg.indexTopk = 2
            cfg.slidingWindow = 16
            cfg.compressRatios = [4]

            let attention = DeepseekV4Attention(config: cfg, layerIdx: 0)
            let cache = DeepseekV4Cache(
                slidingWindow: cfg.slidingWindow,
                compressRatio: 4,
                poolQuantizationEnabled: false)

            // This is the compact state-equivalent of the production
            // 2048 -> 2052 boundary: ratio=4 and topK=2 means eight tokens
            // fill two rows, then token 11 completes row three. Feed every
            // token separately so positions 0...2 also prove that the private
            // indexer branch advances before the first pooled row exists.
            for position in 0..<11 {
                let token = MLXArray.zeros([1, 1, cfg.hiddenSize], dtype: .float16)
                let output = attention(token, mask: .none, cache: cache)
                MLX.eval(output)
                #expect(cache.offset == position + 1)

                let expectedRows = (position + 1) / 4
                #expect((cache.getPooled(.compressor)?.dim(1) ?? 0) == expectedRows)
                #expect((cache.getPooled(.indexer)?.dim(1) ?? 0) == expectedRows)
                #expect(
                    cache.getBuffers(.compressor).kv?.dim(1)
                        == cache.getBuffers(.indexer).kv?.dim(1))
            }

            let boundaryToken = MLXArray.zeros(
                [1, 1, cfg.hiddenSize], dtype: .float16)
            let boundaryOutput = attention(boundaryToken, mask: .none, cache: cache)
            MLX.eval(boundaryOutput)

            #expect(cache.offset == 12)
            #expect(cache.getPooled(.compressor)?.dim(1) == 3)
            #expect(cache.getPooled(.indexer)?.dim(1) == 3)
        }
    }

    @Test("prefill indexer scores mask future compressed chunks before top-k")
    func prefillMasksFutureCompressedChunksBeforeTopK() {
        FocusedMLXTestSupport.withLock {
        // Query position 3 can only see compressed chunk 0 when ratio=4.
        // Chunk 5 has a much larger raw score; if top-k runs before the
        // causal mask, argpartition picks chunk 5 and the later attention
        // visibility mask filters it out, leaving the query starved.
        let scores = MLXArray([
            Float(10), 20, 30, 40, 50, 60,
            Float(10), 20, 30, 40, 50, 60,
            Float(10), 20, 30, 40, 50, 60,
            Float(1), 2, 3, 4, 5, 1000,
        ]).reshaped(1, 4, 6)

        let masked = DeepseekV4Math.causalMaskedIndexerScores(
            scores, offset: 0, ratio: 4)
        let top1 = MLX.argPartition(-masked, kth: 0, axis: -1)[
            .ellipsis, 0..<1
        ]
        MLX.eval(masked, top1)

        #expect(top1[0, 3, 0].item(Int32.self) == 0)
        #expect(masked[0, 3, 0].item(Float.self) > 0)
        #expect(masked[0, 3, 5].item(Float.self) < -1.0e20)
        }
    }

    @Test("ratio-4 overlap cache preserves previous complete window across decode calls")
    func overlapDecodeKeepsPreviousWindowLeftHalf() {
        FocusedMLXTestSupport.withLock {
        var cfg = DeepseekV4Configuration()
        cfg.hiddenSize = 8
        cfg.headDim = 4
        cfg.qkRopeHeadDim = 2
        cfg.rmsNormEps = 1e-6
        let compressor = DeepseekV4Compressor(config: cfg, compressRatio: 4, headDim: 2)
        let cache = DeepseekV4Cache(slidingWindow: 16, compressRatio: 4)

        func tensor(_ start: Int, _ count: Int) -> MLXArray {
            var values: [Float] = []
            for token in start..<(start + count) {
                values.append(Float(token))
                values.append(Float(token) + 0.25)
                values.append(Float(token) + 100.0)
                values.append(Float(token) + 100.25)
            }
            return MLXArray(values).reshaped(1, count, 4)
        }

        // Initial prefill rows 0...7 leave the last complete window
        // (tokens 4...7) in the overlap buffer.
        let prefill = compressor.accumulateOverlapWindows(
            kv: tensor(0, 8),
            gate: tensor(0, 8),
            cache: cache,
            branch: .compressor,
            ratio: 4,
            startPos: 0)
        MLX.eval(prefill.kvRows)
        #expect(prefill.kvRows.shape == [1, 2, 8, 2])
        #expect(prefill.poolBase == 0)

        // Feed decode tokens 8, 9, 10, 11 one at a time. The completed row
        // at token 11 must use tokens 4...7 as its left half and 8...11 as
        // its right half. The old plain remainder-buffer path produced
        // zeros for the left half here.
        for pos in 8..<11 {
            let row = compressor.accumulateOverlapWindows(
                kv: tensor(pos, 1),
                gate: tensor(pos, 1),
                cache: cache,
                branch: .compressor,
                ratio: 4,
                startPos: pos)
            MLX.eval(row.kvRows)
            #expect(row.kvRows.dim(1) == 0)
        }

        let completed = compressor.accumulateOverlapWindows(
            kv: tensor(11, 1),
            gate: tensor(11, 1),
            cache: cache,
            branch: .compressor,
            ratio: 4,
            startPos: 11)
        MLX.eval(completed.kvRows)

        #expect(completed.poolBase == 8)
        #expect(completed.kvRows.shape == [1, 1, 8, 2])
        #expect(completed.kvRows[0, 0, 0, 0].item(Float.self) == 4.0)
        #expect(completed.kvRows[0, 0, 3, 0].item(Float.self) == 7.0)
        #expect(completed.kvRows[0, 0, 4, 0].item(Float.self) == 108.0)
        #expect(completed.kvRows[0, 0, 7, 0].item(Float.self) == 111.0)
        }
    }
}
