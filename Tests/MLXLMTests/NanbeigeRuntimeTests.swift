// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
@testable import MLXLLM
import MLXLMCommon
import Testing

@Suite("Nanbeige 4.2 looped runtime", .serialized)
struct NanbeigeRuntimeTests {
    private static func configuration(runtimeCacheSlots: Int = 4) -> Data {
        """
        {
          "model_type": "nanbeige",
          "vocab_size": 32,
          "hidden_size": 8,
          "intermediate_size": 16,
          "num_hidden_layers": 2,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 4,
          "kv_channels": 4,
          "rms_norm_eps": 1e-5,
          "rope_theta": 70000000,
          "max_position_embeddings": 1024,
          "tie_word_embeddings": false,
          "num_loops": 2,
          "loop_loss_weights": [],
          "skip_loop_final_norm": false,
          "jang_runtime": {
            "architecture": "looped_transformer",
            "cache_layout": "looped_kv_v1",
            "num_loops": 2,
            "num_hidden_layers": 2,
            "cache_slots": \(runtimeCacheSlots),
            "cache_slot_formula": "layer_idx + loop_idx * num_hidden_layers",
            "loop_final_norm": "every_loop",
            "shared_layer_weights_across_loops": true,
            "position_ids_shared_across_loops": true,
            "norm_convention": "llama_rmsnorm_no_plus_one"
          }
        }
        """.data(using: .utf8)!
    }

    @Test("factory registers Nanbeige and the model creates one cache per loop-layer slot")
    func registryAndCacheTopologyUseEffectiveDepth() async throws {
        let config = try JSONDecoder.json5().decode(
            NanbeigeConfiguration.self,
            from: Self.configuration())
        #expect(config.hiddenLayers == 2)
        #expect(config.totalLoops == 2)
        #expect(config.cacheSlots == 4)
        #expect(config.headDimensions == 4)

        let model = NanbeigeModel(config)
        #expect(model.kvHeads == [1, 1, 1, 1])
        #expect(model.newCache(parameters: nil).count == 4)
        #expect(model.newCache(parameters: nil).count == config.runtime?.cacheSlots)

        let created = try await LLMTypeRegistry.shared.createModel(
            configuration: Self.configuration(),
            modelType: "nanbeige")
        #expect(created is NanbeigeModel)
    }

    @Test("every loop owns independent KV slots and all offsets advance together")
    func everyLoopAdvancesItsOwnCacheSlots() throws {
        try MLXMetalTestLock.withLock {
            let config = try JSONDecoder.json5().decode(
                NanbeigeConfiguration.self,
                from: Self.configuration())
            let model = NanbeigeModel(config)
            let cache = model.newCache(parameters: nil)
            let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]

            let logits = model(input, cache: cache)
            MLX.eval(logits)

            #expect(logits.shape == [1, 3, 32])
            #expect(cache.map(\.offset) == [3, 3, 3, 3])
        }
    }

    @Test("loop loss weights override num_loops with count plus one")
    func loopLossWeightsOverrideLoopCount() throws {
        let data = """
        {
          "vocab_size": 32,
          "hidden_size": 8,
          "intermediate_size": 16,
          "num_hidden_layers": 2,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 4,
          "rms_norm_eps": 1e-5,
          "num_loops": 9,
          "loop_loss_weights": [0.25, 0.5],
          "skip_loop_final_norm": false
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder.json5().decode(
            NanbeigeConfiguration.self,
            from: data)
        #expect(config.totalLoops == 3)
        #expect(config.cacheSlots == 6)
    }

    @Test("bundle runtime cache contract mismatch is rejected")
    func mismatchedRuntimeCacheContractIsRejected() {
        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder.json5().decode(
                NanbeigeConfiguration.self,
                from: Self.configuration(runtimeCacheSlots: 2))
        }
    }

    @Test(
        "future Nanbeige architecture flags are rejected instead of silently using the 4.2 path",
        arguments: [
            #""enable_double_loop_split": true"#,
            #""loop_share_kv": true"#,
            #""enable_hyper_connection": true"#,
            #""enable_mhc": true"#,
            #""enable_depth_attention": true"#,
            #""qk_layernorm": true"#,
            #""emb_neighbor_num": 2"#,
        ])
    func unsupportedArchitectureFlagIsRejected(flag: String) {
        let data = """
        {
          "vocab_size": 32,
          "hidden_size": 8,
          "intermediate_size": 16,
          "num_hidden_layers": 2,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 4,
          "rms_norm_eps": 1e-5,
          "num_loops": 2,
          \(flag)
        }
        """.data(using: .utf8)!

        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder.json5().decode(
                NanbeigeConfiguration.self,
                from: data)
        }
    }

    @Test("runtime source uses the non-aliasing loop-layer cache formula")
    func sourceUsesIndependentLoopLayerCacheFormula() throws {
        let sourcePath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Libraries/MLXLLM/Models/Nanbeige.swift")
        let source = try String(contentsOf: sourcePath, encoding: .utf8)

        #expect(source.contains("layerIndex + loopIndex * layers.count"))
        #expect(!source.contains("layerIndex % layers.count"))
        #expect(source.contains("cache == nil || cache?.count == expectedCacheSlots"))
        #expect(source.contains("let mask = createAttentionMask(h: hidden, cache: cache?.first)"))
    }
}
