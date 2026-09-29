import Foundation
import MLXLMCommon

/// Explicit construction hook. The generic factory does not enable this path
/// until mapped decode/prefill and full-model admission have passed their gates.
struct Glm5NextJANGHPreparation {
    let configuration: Glm5NextConfiguration
    let banks: JANGHModelPreparation
    let baseConfiguration: BaseConfiguration

    init(directory: URL, configurationData: Data, sidecar: Data?) throws {
        let config = try JSONDecoder.json5().decode(Glm5NextConfiguration.self, from: configurationData)
        _ = try config.textConfig.validatedSchedule()
        guard !NativeMTPActivation.isExplicitlyRequested || config.textConfig.numNextnPredictLayers == 0 else {
            throw Glm5NextRoutedBankError.incompleteCoverageOrUnsupportedMTP
        }
        let sparse = Set((0 ..< config.textConfig.numHiddenLayers).filter {
            config.textConfig.mlpLayerTypes[$0] == .sparse
        })
        let prepared = try JANGHModelPreparation(
            directory: directory, configuration: configurationData, sidecar: sidecar,
            layout: .init(modelType: "glm5_next", hiddenSize: config.textConfig.hiddenSize,
                          intermediateSize: config.textConfig.moeIntermediateSize,
                          expertCount: config.textConfig.nRoutedExperts, sparseLayers: sparse))
        // Explicit unknown ordinary modes remain errors before any bank mapping.
        baseConfiguration = try JSONDecoder.json5().decode(
            BaseConfiguration.self, from: prepared.ordinaryConfiguration)
        configuration = config
        banks = prepared
    }

    func construct(requesting: Set<ModelRuntimeRequestModality>?) throws -> Glm5Next {
        // Resolve modality admission before mapping the routed banks.
        _ = try Glm5Next.resolveConstruction(configuration, requesting: requesting)
        let routed = try banks.makeRoutedExperts(activationLimit: configuration.textConfig.swigluLimit)
        return try Glm5Next(configuration, requesting: requesting, routedExperts: routed,
                            customRoutedTensorNames: banks.excludedTensorNames)
    }
}
