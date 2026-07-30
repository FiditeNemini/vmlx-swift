// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("Mixed group-size quantization")
struct MixedGroupSizeQuantizationFocusedTests {
    private let bareExpertPath = "layers.1.mlp.switch_mlp.gate_proj"

    @Test("model-prefixed override resolves for a bare module path")
    func modelPrefixedOverrideResolvesForBareModulePath() throws {
        let configJSON = """
            {
              "model_type": "laguna",
              "quantization": {
                "bits": 8,
                "group_size": 64,
                "model.layers.1.mlp.switch_mlp.gate_proj": {
                  "bits": 4,
                  "group_size": 128,
                  "mode": "affine"
                }
              }
            }
            """
        let config = try JSONDecoder().decode(
            BaseConfiguration.self,
            from: Data(configJSON.utf8))

        let bare = config.perLayerQuantization?.quantization(layer: bareExpertPath)
        let prefixed = config.perLayerQuantization?.quantization(
            layer: "model.\(bareExpertPath)")

        #expect(bare?.bits == 4)
        #expect(bare?.groupSize == 128)
        #expect(prefixed == bare)
    }

    @Test("Raptor expert shapes use the module group size during inference")
    func raptorExpertShapesUseModuleGroupSizeDuringInference() {
        let checkpointPath = "model.\(bareExpertPath)"
        let declared = BaseConfiguration.PerLayerQuantization(
            quantization: BaseConfiguration.Quantization(
                groupSize: 64, bits: 8, mode: .affine),
            perLayerQuantization: [
                checkpointPath: .quantize(
                    BaseConfiguration.Quantization(
                        groupSize: 128, bits: 4, mode: .affine))
            ])

        guard let expert = declared.quantization(layer: bareExpertPath) else {
            Issue.record("Expected the Raptor routed-expert quantization override")
            return
        }

        // The real expert tensor is [128, 512, 256] and its scales are
        // [128, 512, 16]. The resolved module group size must establish
        // in_features = scales_last * group_size = 16 * 128 = 2048 and
        // therefore bits = packed_last * 32 / in_features = 4.
        let inferred = JangLoader.inferBitWidthAndGroupSize(
            packedDim: 256,
            numGroups: 16,
            knownGroupSize: expert.groupSize,
            bitWidthsUsed: [4, 8])

        #expect(inferred.bits == 4)
        #expect(inferred.groupSize == 128)
        #expect(16 * inferred.groupSize == 2048)
    }

    @Test("model-prefixed skip is not replaced by the global default")
    func modelPrefixedSkipRemainsSkipped() {
        let settings = BaseConfiguration.PerLayerQuantization(
            quantization: BaseConfiguration.Quantization(groupSize: 64, bits: 8),
            perLayerQuantization: ["model.\(bareExpertPath)": .skip])

        #expect(settings.quantization(layer: bareExpertPath) == nil)
    }
}
