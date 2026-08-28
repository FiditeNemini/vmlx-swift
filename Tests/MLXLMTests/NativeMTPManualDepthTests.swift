import Foundation
import Testing

@testable import MLXLMCommon

/// Manual-depth activation contract (the Speculative Depth 1/2/3 buttons):
/// an explicit user depth activates a tensor-complete MTP head WITHOUT a
/// measured `vmlx_mtp_tuning.json`, Auto stays tuning-gated, and any active
/// launch pairs with enforced greedy sampling for that model+session.
@Suite("Native MTP manual depth contract")
struct NativeMTPManualDepthTests {

    private static func config(modelType: String = "qwen4_exp") -> Data {
        Data("{\"model_type\": \"\(modelType)\"}".utf8)
    }

    /// A Flash-Next-shaped bundle: complete 57-tensor head, no tuning file.
    private static var completeUntunedStatus: MTPBundleStatus {
        MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,
            tensorCount: 57,
            mode: .preservedEnabled,
            nativeMTPTuning: nil)
    }

    /// JANG_1L-shaped: config declares a layer, bundle has zero tensors.
    private static var incompleteStatus: MTPBundleStatus {
        MTPBundleStatus(
            bundleHasMTP: false,
            configuredLayers: 1,
            tensorCount: 0,
            mode: .metadataOnlyMissingWeights,
            nativeMTPTuning: nil)
    }

    private static func settings(
        mode: VMLXMTPServerMode, explicitDepth: Int? = nil
    ) -> VMLXServerRuntimeSettings {
        var settings = VMLXServerRuntimeSettings()
        settings.mtp.mode = mode
        settings.mtp.explicitDepth = explicitDepth
        return settings
    }

    @Test("manual depth activates a complete untuned head at exactly that depth")
    func manualDepthActivatesWithoutTuning() throws {
        for depth in 1...3 {
            let settings = Self.settings(mode: .forceOn, explicitDepth: depth)
            #expect(
                settings.effectiveMTPLaunchMode(for: Self.completeUntunedStatus)
                    == .speculative)
            let launch = settings.resolvedMTPLaunch(
                configData: Self.config(),
                jangConfig: nil,
                status: Self.completeUntunedStatus)
            #expect(launch.launchMode == .speculative)
            #expect(launch.recommendation?.depth == depth)
        }
    }

    @Test("auto stays tuning-gated for the same complete untuned head")
    func autoStaysTuningGated() throws {
        let settings = Self.settings(mode: .auto)
        #expect(
            settings.effectiveMTPLaunchMode(for: Self.completeUntunedStatus) == .off)
        let launch = settings.resolvedMTPLaunch(
            configData: Self.config(),
            jangConfig: nil,
            status: Self.completeUntunedStatus)
        #expect(launch.launchMode == .off)
    }

    @Test("manual depth fails closed on an incomplete artifact (JANG_1L shape)")
    func manualDepthFailsClosedWithoutTensors() throws {
        let settings = Self.settings(mode: .forceOn, explicitDepth: 2)
        #expect(
            settings.effectiveMTPLaunchMode(for: Self.incompleteStatus) == .blocked)
        let launch = settings.resolvedMTPLaunch(
            configData: Self.config(),
            jangConfig: nil,
            status: Self.incompleteStatus)
        #expect(launch.launchMode == .blocked)
        #expect(
            settings.validationIssues(mtpStatus: Self.incompleteStatus)
                .contains { $0.severity == .error && $0.field == "mtp.mode" })
    }

    @Test("manual depth rejects unsupported families and bad depths")
    func manualDepthRejectsUnsupportedAndBadDepth() throws {
        let unsupported = Self.settings(mode: .forceOn, explicitDepth: 2)
        let launch = unsupported.resolvedMTPLaunch(
            configData: Self.config(modelType: "llama"),
            jangConfig: nil,
            status: Self.completeUntunedStatus)
        #expect(launch.launchMode == .blocked)

        let badDepth = Self.settings(mode: .forceOn, explicitDepth: 4)
        #expect(
            badDepth.validationIssues(mtpStatus: Self.completeUntunedStatus)
                .contains { $0.field == "mtp.explicitDepth" })
        let badLaunch = badDepth.resolvedMTPLaunch(
            configData: Self.config(),
            jangConfig: nil,
            status: Self.completeUntunedStatus)
        #expect(badLaunch.launchMode == .blocked)
    }

    @Test("the load gate accepts a manual-depth run without usable tuning")
    func loadGateAcceptsManualDepth() throws {
        try NativeMTPActivation.$explicitRequestOverride.withValue(true) {
            try NativeMTPActivation.$manualDepthOverride.withValue(2) {
                let allowed = try NativeMTPActivation.shouldLoadNativeMTPWeights(
                    configData: Self.config(),
                    baseModelType: "qwen4_exp",
                    status: Self.completeUntunedStatus)
                #expect(allowed)
            }
        }
        // Without the manual override the same untuned bundle refuses.
        try NativeMTPActivation.$explicitRequestOverride.withValue(true) {
            #expect(throws: (any Error).self) {
                _ = try NativeMTPActivation.shouldLoadNativeMTPWeights(
                    configData: Self.config(),
                    baseModelType: "qwen4_exp",
                    status: Self.completeUntunedStatus)
            }
        }
    }

    @Test("enforced greedy sampling constants are exact")
    func enforcedGreedySamplingConstants() {
        let greedy = VMLXServerMTPSettings.mtpEnforcedGreedySampling
        #expect(greedy.temperature == 0)
        #expect(greedy.topP == 1)
        #expect(greedy.topK == 0)
        #expect(greedy.minP == 0)
    }
}
