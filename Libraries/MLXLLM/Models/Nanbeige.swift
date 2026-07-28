// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Nanbeige 4.2 is a looped Transformer: one physical stack of Llama-shaped
/// layers is executed repeatedly with independent KV state for every
/// (loop, layer) pair.
public struct NanbeigeConfiguration: Codable, Sendable {
    let hiddenSize: Int
    let hiddenLayers: Int
    let intermediateSize: Int
    let attentionHeads: Int
    let kvHeads: Int
    let headDimensions: Int
    let kvChannels: Int?
    let rmsNormEps: Float
    let vocabularySize: Int
    let maxPositionEmbeddings: Int?
    let ropeTheta: Float
    let tieWordEmbeddings: Bool
    let attentionBias: Bool
    let mlpBias: Bool

    let numLoops: Int
    let loopLossWeights: [Float]?
    let skipLoopFinalNorm: Bool

    let enableDoubleLoopSplit: Bool?
    let loopShareKV: Bool?
    let enableHyperConnection: Bool?
    let enableMHC: Bool?
    let enableDepthAttention: Bool?
    let qkLayerNorm: Bool?
    let embeddingNeighborCount: Int?

    let runtime: RuntimeContract?

    var totalLoops: Int {
        if let loopLossWeights, !loopLossWeights.isEmpty {
            return loopLossWeights.count + 1
        }
        return max(1, numLoops)
    }

    var cacheSlots: Int {
        hiddenLayers * totalLoops
    }

    struct RuntimeContract: Codable, Sendable {
        let architecture: String
        let cacheLayout: String
        let numLoops: Int
        let hiddenLayers: Int
        let cacheSlots: Int
        let cacheSlotFormula: String
        let loopFinalNorm: String
        let sharedLayerWeights: Bool
        let sharedPositionIDs: Bool
        let normConvention: String

        enum CodingKeys: String, CodingKey {
            case architecture
            case cacheLayout = "cache_layout"
            case numLoops = "num_loops"
            case hiddenLayers = "num_hidden_layers"
            case cacheSlots = "cache_slots"
            case cacheSlotFormula = "cache_slot_formula"
            case loopFinalNorm = "loop_final_norm"
            case sharedLayerWeights = "shared_layer_weights_across_loops"
            case sharedPositionIDs = "position_ids_shared_across_loops"
            case normConvention = "norm_convention"
        }
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDimensions = "head_dim"
        case kvChannels = "kv_channels"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case numLoops = "num_loops"
        case loopLossWeights = "loop_loss_weights"
        case skipLoopFinalNorm = "skip_loop_final_norm"
        case enableDoubleLoopSplit = "enable_double_loop_split"
        case loopShareKV = "loop_share_kv"
        case enableHyperConnection = "enable_hyper_connection"
        case enableMHC = "enable_mhc"
        case enableDepthAttention = "enable_depth_attention"
        case qkLayerNorm = "qk_layernorm"
        case embeddingNeighborCount = "emb_neighbor_num"
        case runtime = "jang_runtime"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        let explicitHeadDimensions = try container.decodeIfPresent(
            Int.self, forKey: .headDimensions)
        kvChannels = try container.decodeIfPresent(Int.self, forKey: .kvChannels)
        if let headDimensions = explicitHeadDimensions ?? kvChannels {
            self.headDimensions = headDimensions
        } else {
            throw DecodingError.keyNotFound(
                CodingKeys.headDimensions,
                DecodingError.Context(
                    codingPath: container.codingPath,
                    debugDescription:
                        "Nanbeige requires head_dim or kv_channels; hidden_size / heads is invalid"))
        }
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false

        numLoops = try container.decodeIfPresent(Int.self, forKey: .numLoops) ?? 1
        loopLossWeights = try container.decodeIfPresent([Float].self, forKey: .loopLossWeights)
        skipLoopFinalNorm =
            try container.decodeIfPresent(Bool.self, forKey: .skipLoopFinalNorm) ?? false

        enableDoubleLoopSplit = try container.decodeIfPresent(
            Bool.self, forKey: .enableDoubleLoopSplit)
        loopShareKV = try container.decodeIfPresent(Bool.self, forKey: .loopShareKV)
        enableHyperConnection = try container.decodeIfPresent(
            Bool.self, forKey: .enableHyperConnection)
        enableMHC = try container.decodeIfPresent(Bool.self, forKey: .enableMHC)
        enableDepthAttention = try container.decodeIfPresent(
            Bool.self, forKey: .enableDepthAttention)
        qkLayerNorm = try container.decodeIfPresent(Bool.self, forKey: .qkLayerNorm)
        embeddingNeighborCount = try container.decodeIfPresent(
            Int.self, forKey: .embeddingNeighborCount)
        runtime = try container.decodeIfPresent(RuntimeContract.self, forKey: .runtime)

        try validate(container: container)
    }

    private func validate(
        container: KeyedDecodingContainer<CodingKeys>
    ) throws {
        func reject(_ condition: Bool, _ key: CodingKeys, _ description: String) throws {
            if condition {
                throw DecodingError.dataCorruptedError(
                    forKey: key, in: container, debugDescription: description)
            }
        }

        try reject(
            enableDoubleLoopSplit == true, .enableDoubleLoopSplit,
            "Nanbeige double-loop split is not implemented")
        try reject(
            loopShareKV == true, .loopShareKV,
            "Nanbeige shared-KV loops are not compatible with independent loop slots")
        try reject(
            enableHyperConnection == true, .enableHyperConnection,
            "Nanbeige hyper-connections are not implemented")
        try reject(enableMHC == true, .enableMHC, "Nanbeige mHC is not implemented")
        try reject(
            enableDepthAttention == true, .enableDepthAttention,
            "Nanbeige depth attention is not implemented")
        try reject(qkLayerNorm == true, .qkLayerNorm, "Nanbeige QK layer norm is not implemented")
        try reject(
            (embeddingNeighborCount ?? 0) != 0, .embeddingNeighborCount,
            "Nanbeige n-gram neighbor embeddings are not implemented")
        try reject(numLoops < 1, .numLoops, "Nanbeige num_loops must be positive")

        if let runtime {
            try reject(
                runtime.architecture != "looped_transformer", .runtime,
                "Nanbeige jang_runtime architecture must be looped_transformer")
            try reject(
                runtime.cacheLayout != "looped_kv_v1", .runtime,
                "Nanbeige jang_runtime cache_layout must be looped_kv_v1")
            try reject(
                runtime.numLoops != totalLoops, .runtime,
                "Nanbeige jang_runtime num_loops does not match the effective loop count")
            try reject(
                runtime.hiddenLayers != hiddenLayers, .runtime,
                "Nanbeige jang_runtime num_hidden_layers does not match config")
            try reject(
                runtime.cacheSlots != cacheSlots, .runtime,
                "Nanbeige jang_runtime cache_slots must equal layers * loops")
            try reject(
                runtime.cacheSlotFormula != "layer_idx + loop_idx * num_hidden_layers", .runtime,
                "Nanbeige jang_runtime cache_slot_formula is unsupported")
            try reject(
                runtime.loopFinalNorm
                    != (skipLoopFinalNorm ? "after_all_loops" : "every_loop"),
                .runtime,
                "Nanbeige jang_runtime loop_final_norm does not match skip_loop_final_norm")
            try reject(
                !runtime.sharedLayerWeights, .runtime,
                "Nanbeige requires shared layer weights across loops")
            try reject(
                !runtime.sharedPositionIDs, .runtime,
                "Nanbeige requires shared positions across loops")
            try reject(
                runtime.normConvention != "llama_rmsnorm_no_plus_one", .runtime,
                "Nanbeige requires plain Llama RMSNorm weights")
        }
    }

    var llamaConfiguration: LlamaConfiguration {
        LlamaConfiguration(
            hiddenSize: hiddenSize,
            hiddenLayers: hiddenLayers,
            intermediateSize: intermediateSize,
            attentionHeads: attentionHeads,
            headDimensions: headDimensions,
            rmsNormEps: rmsNormEps,
            vocabularySize: vocabularySize,
            kvHeads: kvHeads,
            maxPositionEmbeddings: maxPositionEmbeddings,
            ropeTheta: ropeTheta,
            ropeTraditional: false,
            tieWordEmbeddings: tieWordEmbeddings,
            attentionBias: attentionBias,
            mlpBias: mlpBias)
    }
}

public final class NanbeigeModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [LlamaTransformerBlock]
    let norm: RMSNorm
    let totalLoops: Int
    let skipLoopFinalNorm: Bool

    init(_ configuration: NanbeigeConfiguration) {
        precondition(configuration.vocabularySize > 0)
        let llama = configuration.llamaConfiguration
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize,
            dimensions: configuration.hiddenSize)
        layers = (0 ..< configuration.hiddenLayers).map { _ in
            LlamaTransformerBlock(llama)
        }
        norm = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps)
        totalLoops = configuration.totalLoops
        skipLoopFinalNorm = configuration.skipLoopFinalNorm
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hidden = embedTokens(inputs)
        let expectedCacheSlots = layers.count * totalLoops
        precondition(
            cache == nil || cache?.count == expectedCacheSlots,
            "nanbeige: expected \(expectedCacheSlots) cache slots, got \(cache?.count ?? 0)")

        // Every loop processes the same token positions. Compute the mask once,
        // before any loop advances its independent cache.
        let mask = createAttentionMask(h: hidden, cache: cache?.first)

        for loopIndex in 0 ..< totalLoops {
            for (layerIndex, layer) in layers.enumerated() {
                let cacheIndex = layerIndex + loopIndex * layers.count
                hidden = layer(hidden, mask: mask, cache: cache?[cacheIndex])
            }
            if !skipLoopFinalNorm {
                hidden = norm(hidden)
            }
        }
        if skipLoopFinalNorm {
            hidden = norm(hidden)
        }
        return hidden
    }
}

public final class NanbeigeModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let model: NanbeigeModelInner
    let configuration: NanbeigeConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ configuration: NanbeigeConfiguration) {
        let vocabularySize = configuration.vocabularySize
        let hiddenSize = configuration.hiddenSize
        let cacheSlots = configuration.cacheSlots
        let runtimeCacheSlots = configuration.runtime?.cacheSlots
        let tiedWordEmbeddings = configuration.tieWordEmbeddings

        self.configuration = configuration
        self.vocabularySize = vocabularySize
        kvHeads = Array(
            repeating: configuration.kvHeads,
            count: cacheSlots)
        model = NanbeigeModelInner(configuration)
        if !tiedWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                hiddenSize,
                vocabularySize,
                bias: false)
        }
        precondition(
            runtimeCacheSlots == nil || runtimeCacheSlots == kvHeads.count,
            "nanbeige: bundle cache_slots does not match runtime cache topology")
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let output = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(output)
        }
        return model.embedTokens.asLinear(output)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.contains("self_attn.rotary_emb.inv_freq") }
    }

    public func messageGenerator(tokenizer: any Tokenizer) -> any MessageGenerator {
        do {
            _ = try tokenizer.applyChatTemplate(messages: [
                ["role": "system", "content": "test"]
            ])
            return DefaultMessageGenerator()
        } catch {
            return NoSystemMessageGenerator()
        }
    }
}

extension NanbeigeModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
