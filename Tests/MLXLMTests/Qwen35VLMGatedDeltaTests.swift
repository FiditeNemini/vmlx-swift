import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import Testing

@Suite("Qwen3.5-VL gated-delta forward", .serialized)
struct Qwen35VLMGatedDeltaTests {
    private func tinyConfiguration() throws -> Qwen35Configuration {
        try JSONDecoder.json5().decode(
            Qwen35Configuration.self,
            from: """
            {
              "model_type": "qwen3_5_moe",
              "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 32,
                "num_hidden_layers": 4,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 2,
                "head_dim": 8,
                "full_attention_interval": 4,
                "vocab_size": 100,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                  "rope_type": "default",
                  "rope_theta": 100000.0,
                  "partial_rotary_factor": 0.25,
                  "mrope_section": [1, 1, 1]
                }
              },
              "vision_config": {
                "model_type": "qwen3_vl",
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "out_hidden_size": 32,
                "num_heads": 4,
                "patch_size": 2,
                "spatial_merge_size": 2,
                "temporal_patch_size": 1,
                "num_position_embeddings": 32
              },
              "vocab_size": 100,
              "image_token_id": 98,
              "video_token_id": 97
            }
            """.data(using: .utf8)!)
    }

    @Test("tiny VLM text path executes linear-attention gated-delta cache")
    func tinyVLMTextPathExecutesGatedDeltaCache() throws {
        try MLXMetalTestLock.withLock {
            let model = Qwen35(try tinyConfiguration())
            let cache = model.newCache(parameters: nil)
            let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]

            let logits = model(input, cache: cache)
            MLX.eval(logits)

            #expect(logits.shape == [1, 3, 100])
            #expect(cache.count == 4)
            #expect(cache.first is MambaCache)
            #expect(cache.first?.offset == 3)
            let mambaCache = try #require(cache.first as? MambaCache)
            #expect(mambaCache[1]?.dtype == .float32)
        }
    }

    @Test("batch=2 decode preserves one position offset per sequence")
    func batchDecodePreservesPerSequenceOffsets() throws {
        try MLXMetalTestLock.withLock {
            let model = Qwen35(try tinyConfiguration())
            let first = model.newCache(parameters: nil)
            let second = model.newCache(parameters: nil)

            MLX.eval(model(MLXArray([1, 2, 3])[.newAxis, .ellipsis], cache: first))
            MLX.eval(model(MLXArray([4, 5])[.newAxis, .ellipsis], cache: second))

            let batchCaches: [KVCache] = zip(first, second).map { lhs, rhs in
                if let lhs = lhs as? ArraysCache, let rhs = rhs as? ArraysCache {
                    return BatchArraysCache(slotCaches: [lhs, rhs])
                }
                return BatchKVCache(slotCaches: [lhs, rhs])
            }
            let decodeTokens = MLXArray([Int32(6), Int32(7)]).reshaped(2, 1)
            let logits = model(decodeTokens, cache: batchCaches)
            MLX.eval(logits)

            #expect(logits.shape == [2, 1, 100])
            let fullAttention = try #require(batchCaches.last as? BatchKVCache)
            #expect(fullAttention.offsetArray.asArray(Int32.self) == [4, 3])
        }
    }
}
