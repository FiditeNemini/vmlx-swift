// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXVLM

@Suite("qwen4_exp configuration")
struct Qwen4ExpConfigurationTests {
    private func configData(dtype: String) -> Data {
        Data("""
            {
              "model_type":"qwen4_exp",
              "text_config":{
                "model_type":"qwen4_exp_text","dtype":"\(dtype)",
                "mamba_ssm_dtype":"float32","mtp_num_hidden_layers":1,
                "hidden_size":64,"num_hidden_layers":2,"intermediate_size":64,
                "num_attention_heads":4,"num_key_value_heads":1,"head_dim":16,
                "linear_num_value_heads":4,"linear_num_key_heads":1,
                "linear_key_head_dim":16,"linear_value_head_dim":16,
                "linear_conv_kernel_dim":4,"vocab_size":128,
                "num_experts":8,"num_experts_per_tok":2,
                "moe_intermediate_size":16,"shared_expert_intermediate_size":16,
                "layer_types":["linear_attention","full_attention"],
                "hc_count":4,"hc_lowrank":8,"ple_layer_ids":[2],
                "ple_embed_dim":64,"ple_conv_kernel_size":4,
                "ngram_size":3,"heads_per_ngram":2,"ngram_vocab_size_base":101,
                "make_ngram_vocab_size_divisible_by":128,"seed":null,
                "split_ngram_parts":4,"indexer_n_heads":2,"indexer_kv_heads":1,
                "indexer_head_dim":8,"indexer_budget":32,"indexer_compress_ratio":4
              },
              "vision_config":{
                "model_type":"qwen3_vl","depth":2,"hidden_size":64,
                "intermediate_size":128,"out_hidden_size":64,"num_heads":4,
                "patch_size":14,"spatial_merge_size":2,"temporal_patch_size":2,
                "num_position_embeddings":64
              }
            }
            """.utf8)
    }

    @Test("decodes native text_config fields and null seed fallback")
    func decodesNativeConfig() throws {
        let config = try JSONDecoder().decode(
            Qwen4ExpConfiguration.self, from: configData(dtype: "bfloat16"))
        #expect(config.base.textConfiguration.modelType == "qwen4_exp_text")
        #expect(config.declaredComputeDType == .bfloat16)
        #expect(config.extras.mambaSSMDType == "float32")
        #expect(config.extras.mtpNumHiddenLayers == 1)
        #expect(config.extras.layerTypes == ["linear_attention", "full_attention"])
        #expect(config.extras.seed == 1234)
        #expect(config.extras.hcCount == 4)
        #expect(VLMTypeRegistry.supportedModelTypes.contains("qwen4_exp"))

        let model = Qwen4Exp(config)
        #expect(model.usesResidentNativeAffineRoutedExperts)
        #expect(model.requiresExactTensorMmapBuffers)
        #expect(!model.excludeFromGenericSafetensorsLoad(
            key: "language_model.layers.0.mlp.switch_mlp.gate_proj.weight"))
        #expect(model.excludeFromGenericSafetensorsLoad(
            key: "language_model.layers.1.ple.ngram_embedding.shards.0.weight"))
        #expect(!model.excludeFromGenericSafetensorsLoad(
            key: "language_model.layers.0.mlp.shared_expert.gate_proj.weight"))

        let sanitized = model.sanitize(weights: [
            "language_model.layers.0.mlp.switch_mlp.gate_proj.weight":
                MLXArray.zeros([2, 2], dtype: .uint32),
            "language_model.layers.0.mlp.switch_mlp.gate_proj.scales":
                MLXArray.ones([2, 2], dtype: .float16),
            "language_model.layers.0.mlp.switch_mlp.gate_proj.biases":
                MLXArray.zeros([2, 2], dtype: .float16),
            "language_model.layers.0.attn_hyper_connection.hc_norm.weight":
                MLXArray.ones([64], dtype: .bfloat16),
            "language_model.layers.0.linear_attn.A_log":
                MLXArray.ones([4], dtype: .float32),
        ])
        #expect(sanitized[
            "language_model.layers.0.mlp.switch_mlp.gate_proj.scales"]?.dtype == .float16)
        #expect(sanitized[
            "language_model.layers.0.mlp.switch_mlp.gate_proj.biases"]?.dtype == .float16)
        #expect(sanitized[
            "language_model.layers.0.attn_hyper_connection.hc_norm.weight"]?.dtype == .bfloat16)
        #expect(sanitized["language_model.layers.0.linear_attn.A_log"]?.dtype == .float32)

    }

    @Test("4M q4g64 verifier isolates the complete MoE block by decode row")
    func q4VerifierUsesDecodeEquivalentMoERows() throws {
        let config = try JSONDecoder().decode(
            Qwen4ExpConfiguration.self, from: configData(dtype: "bfloat16"))
        var text = config.base.textConfiguration
        text.moeIntermediateSize = 64
        text.sharedExpertIntermediateSize = 64
        let block = Qwen35Language.SparseMoeBlock(
            text, layerIdx: 0,
            allowFusedGateUpCache: false, compileDecodeRegions: true)
        quantize(model: block, groupSize: 64, bits: 4)

        let values = (0 ..< (4 * 64)).map { Float(($0 % 37) - 18) / 19 }
        let input = MLXArray(values).reshaped(1, 4, 64).asType(.bfloat16)
        #expect(block.requiresDecodeEquivalentRows(input))

        let batched = block(input)
        let rows = MLX.split(input, parts: 4, axis: 1)
        let independent = MLX.concatenated(rows.map { block($0) }, axis: 1)
        let error = max(
            abs(batched.asType(.float32) - independent.asType(.float32)))
        MLX.eval(error)
        #expect(error.item(Float.self) == 0)

        for bits in [2, 6] {
            let nonFourBit = Qwen35Language.SparseMoeBlock(
                text, layerIdx: 0,
                allowFusedGateUpCache: false, compileDecodeRegions: true)
            quantize(model: nonFourBit, groupSize: 64, bits: bits)
            #expect(!nonFourBit.requiresDecodeEquivalentRows(input))
        }
    }

    @Test("packed affine metadata preserves checkpoint storage dtype")
    func packedAffineMetadataPreservesStorageDType() throws {
        let config = try JSONDecoder().decode(
            Qwen4ExpConfiguration.self, from: configData(dtype: "float16"))
        let model = Qwen4Exp(config)
        let sanitized = model.sanitize(weights: [
            "language_model.embed_tokens.weight":
                MLXArray.zeros([2, 2], dtype: .uint32),
            "language_model.embed_tokens.scales":
                MLXArray.ones([2, 2], dtype: .bfloat16),
            "language_model.embed_tokens.biases":
                MLXArray.zeros([2, 2], dtype: .bfloat16),
            "lm_head.weight": MLXArray.zeros([2, 2], dtype: .uint32),
            "lm_head.scales": MLXArray.ones([2, 2], dtype: .bfloat16),
            "lm_head.biases": MLXArray.zeros([2, 2], dtype: .bfloat16),
            "language_model.layers.0.linear_attn.A_log":
                MLXArray.ones([4], dtype: .float32),
        ])
        #expect(sanitized["language_model.embed_tokens.scales"]?.dtype == .bfloat16)
        #expect(sanitized["language_model.embed_tokens.biases"]?.dtype == .bfloat16)
        #expect(sanitized["lm_head.scales"]?.dtype == .bfloat16)
        #expect(sanitized["lm_head.biases"]?.dtype == .bfloat16)
        #expect(sanitized["language_model.layers.0.linear_attn.A_log"]?.dtype == .float32)
    }
}
