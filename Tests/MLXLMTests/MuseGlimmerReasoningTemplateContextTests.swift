//
//  MuseGlimmerReasoningTemplateContextTests.swift
//  MLXLMTests
//
//  Muse Glimmer's template keys reasoning on `reasoning_strength`, not the
//  `reasoning_effort` the request surface carries, and defaults to `high` when
//  the variable is undefined. Nothing in the library ever set that key, so
//  every request ran at `high` no matter what the caller chose — an editable,
//  saved, inert control.
//

import Foundation
import MLX
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

@Suite("Muse Glimmer reasoning translation")
struct MuseGlimmerReasoningTemplateContextTests {

    private static func apply(
        _ context: [String: any Sendable],
        modelType: String = "muse_glimmer"
    ) -> [String: any Sendable]? {
        MuseGlimmerReasoningTemplateContext.apply(
            additionalContext: context, modelType: modelType)
    }

    @Test("reasoning_effort becomes reasoning_strength")
    func translatesEffort() throws {
        for (effort, expected) in [
            ("low", "low"), ("medium", "medium"), ("high", "high"), ("xhigh", "xhigh"),
        ] {
            let out = try #require(Self.apply(["reasoning_effort": effort]))
            #expect(out["reasoning_strength"] as? String == expected, "effort \(effort)")
        }
    }

    /// The level set is FOUR. A caller whose ceiling is `max` must reach
    /// `xhigh`, not be flattened into `high` — that silently caps the model
    /// below its intended ceiling for exactly the coding/agentic work the
    /// model card says `xhigh` is for.
    @Test("max reaches xhigh rather than capping at high")
    func maxReachesXhigh() throws {
        let out = try #require(Self.apply(["reasoning_effort": "max"]))
        #expect(out["reasoning_strength"] as? String == "xhigh")
    }

    /// There is no reasoning channel — the strength is a system-prompt
    /// sentence, so there are no think tags to suppress. "Off" must therefore
    /// map to the floor. Removing the key instead would fall through to the
    /// template's `high` default: the exact opposite of the request.
    @Test("disabling maps to the floor, never to an absent key")
    func offMapsToFloor() throws {
        for off in ["none", "off", "minimal", "no_think"] {
            let out = try #require(Self.apply(["reasoning_effort": off]))
            #expect(out["reasoning_strength"] as? String == "low", "value \(off)")
        }
        let disabled = try #require(Self.apply(["enable_thinking": false]))
        #expect(disabled["reasoning_strength"] as? String == "low")
        let enabled = try #require(Self.apply(["enable_thinking": true]))
        #expect(enabled["reasoning_strength"] as? String == "high")
    }

    /// An explicit `reasoning_strength` is already in the template's own
    /// vocabulary and must win — re-deriving it from a generic
    /// `reasoning_effort` would let a default override a deliberate choice.
    @Test("an explicit reasoning_strength is not overridden by reasoning_effort")
    func explicitStrengthWins() throws {
        let out = try #require(
            Self.apply(["reasoning_strength": "xhigh", "reasoning_effort": "low"]))
        #expect(out["reasoning_strength"] as? String == "xhigh")
    }

    /// With nothing requested, leave the key unset so the template's own
    /// documented `high` default applies. Injecting a value here would be this
    /// adapter inventing a policy.
    @Test("no request leaves the key unset for the template default")
    func noRequestLeavesUnset() {
        let out = MuseGlimmerReasoningTemplateContext.apply(
            additionalContext: ["add_generation_prompt": true], modelType: "muse_glimmer")
        #expect(out?["reasoning_strength"] == nil)
    }

    /// The gate is the whole safety story: every other family must pass
    /// through byte-identical, or this adapter would corrupt their contracts.
    @Test("other families are untouched")
    func otherFamiliesUntouched() throws {
        for other in ["hy_v3", "deepseek_v4", "gemma4", "qwen3_5", "gptoss", "apertus1p5"] {
            let input: [String: any Sendable] = ["reasoning_effort": "low"]
            let out = try #require(
                MuseGlimmerReasoningTemplateContext.apply(
                    additionalContext: input, modelType: other))
            #expect(out["reasoning_strength"] == nil, "\(other) gained a Muse key")
            #expect(out["reasoning_effort"] as? String == "low", "\(other) lost its effort")
        }
    }

    @Test("the model-type gate accepts the shipped variants")
    func gateMatchesVariants() {
        #expect(MuseGlimmerReasoningTemplateContext.applies(to: "muse_glimmer"))
        #expect(MuseGlimmerReasoningTemplateContext.applies(to: "muse_glimmer_text"))
        #expect(MuseGlimmerReasoningTemplateContext.applies(to: "MUSE_GLIMMER"))
        #expect(!MuseGlimmerReasoningTemplateContext.applies(to: "muse"))
        #expect(!MuseGlimmerReasoningTemplateContext.applies(to: nil))
    }

    // MARK: - Reachability

    /// Records the context that actually reached the underlying processor.
    private final class SpyProcessor: UserInputProcessor, @unchecked Sendable {
        var seen: [String: any Sendable]?
        func prepare(input: UserInput) throws -> LMInput {
            seen = input.additionalContext
            return LMInput(text: .init(tokens: MLXArray([0])))
        }
    }

    /// THE WIRING, not the adapter. Muse Glimmer declares `supports_thinking`,
    /// so `VLMDefaultContextUserInputProcessor.defaultContext` returns nil for
    /// it — and the wrapper's old `guard let defaultAdditionalContext` early
    /// return meant a translation placed inside it would never run. A correct
    /// adapter that is never reached is the failure mode this pins.
    @Test("the translation reaches the processor even with no default context")
    func translationIsReachedWithoutDefaults() async throws {
        let spy = SpyProcessor()
        let wrapper = VLMDefaultContextUserInputProcessor(
            base: spy,
            defaultAdditionalContext: nil,          // exactly Muse's situation
            modelType: "muse_glimmer"
        )
        _ = try await wrapper.prepare(
            input: UserInput(
                prompt: .text("hi"), additionalContext: ["reasoning_effort": "low"]))
        let seen = try #require(spy.seen, "the base processor was never called")
        #expect(
            seen["reasoning_strength"] as? String == "low",
            "reasoning_strength never reached the template")
    }

    /// The same wrapper must leave a non-Muse model's context alone.
    @Test("the wrapper does not inject a Muse key for other families")
    func wrapperLeavesOtherFamiliesAlone() async throws {
        let spy = SpyProcessor()
        let wrapper = VLMDefaultContextUserInputProcessor(
            base: spy, defaultAdditionalContext: nil, modelType: "gemma4")
        _ = try await wrapper.prepare(
            input: UserInput(
                prompt: .text("hi"), additionalContext: ["reasoning_effort": "low"]))
        // With no defaults and no translation there is nothing to merge, so the
        // wrapper hands the ORIGINAL input straight through.
        let seen = try #require(spy.seen)
        #expect(seen["reasoning_strength"] == nil)
        #expect(seen["reasoning_effort"] as? String == "low")
    }
}
