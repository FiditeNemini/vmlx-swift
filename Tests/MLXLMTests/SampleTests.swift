// Copyright © 2025 Apple Inc.

import MLX
import MLXLMCommon
import XCTest

public class SampleTests: XCTestCase {

    private func sampleCounts(sampler: TopPSampler, logits: MLXArray, draws: Int) -> [Int: Int] {
        var counts: [Int: Int] = [:]
        for _ in 0 ..< draws {
            let token = sampler.sample(logits: logits).item(Int.self)
            counts[token, default: 0] += 1
        }
        return counts
    }

    private func frequency(_ counts: [Int: Int], token: Int, draws: Int) -> Float {
        Float(counts[token, default: 0]) / Float(draws)
    }

    private func assertOnlySampled(
        _ counts: [Int: Int], allowedTokens: Set<Int>, file: StaticString = #filePath,
        line: UInt = #line
    ) {
        for token in counts.keys {
            XCTAssertTrue(
                allowedTokens.contains(token), "Unexpected sampled token: \(token)", file: file,
                line: line)
        }
    }

    func testTopKSamplerKeepsOnlyTopToken() {
        let sampler = TopPSampler(temperature: 1.0, topK: 1)
        let logits = MLXArray([0.1 as Float, 2.0 as Float, 1.0 as Float])[.newAxis, .ellipsis]

        for _ in 0 ..< 10 {
            let token = sampler.sample(logits: logits).item(Int.self)
            XCTAssertEqual(token, 1)
        }
    }

    func testTopPSamplerLowThresholdKeepsMaxToken() {
        let probs = MLXArray([0.9 as Float, 0.0 as Float, 0.0 as Float, 0.1 as Float])[
            .newAxis, .ellipsis]
        let sampler = TopPSampler(temperature: 1.0, topP: 0.3)
        let counts = sampleCounts(sampler: sampler, logits: log(probs), draws: 200)

        XCTAssertEqual(counts[0], 200)
        assertOnlySampled(counts, allowedTokens: [0])
    }

    func testTopPSamplerPartialMassKeepsExpectedDistribution() {
        let probs = MLXArray([0.0 as Float, 0.5 as Float, 0.4 as Float, 0.1 as Float])[
            .newAxis, .ellipsis]
        let draws = 4000
        let sampler = TopPSampler(temperature: 1.0, topP: 0.6)
        let counts = sampleCounts(sampler: sampler, logits: log(probs), draws: draws)

        assertOnlySampled(counts, allowedTokens: [1, 2])
        XCTAssertEqual(frequency(counts, token: 1, draws: draws), 0.5556, accuracy: 0.06)
        XCTAssertEqual(frequency(counts, token: 2, draws: draws), 0.4444, accuracy: 0.06)
    }

    func testTopPSamplerHighThresholdKeepsExpectedDistribution() {
        let probs = MLXArray([0.0 as Float, 0.5 as Float, 0.4 as Float, 0.1 as Float])[
            .newAxis, .ellipsis]
        let draws = 4000
        let sampler = TopPSampler(temperature: 1.0, topP: 0.95)
        let counts = sampleCounts(sampler: sampler, logits: log(probs), draws: draws)

        assertOnlySampled(counts, allowedTokens: [1, 2, 3])
        XCTAssertEqual(frequency(counts, token: 1, draws: draws), 0.5, accuracy: 0.06)
        XCTAssertEqual(frequency(counts, token: 2, draws: draws), 0.4, accuracy: 0.06)
        XCTAssertEqual(frequency(counts, token: 3, draws: draws), 0.1, accuracy: 0.04)
    }

    func testTopKSamplerTopTwoKeepsExpectedDistribution() {
        let probs = MLXArray([0.6 as Float, 0.0 as Float, 0.1 as Float, 0.3 as Float])[
            .newAxis, .ellipsis]
        let draws = 4000
        let sampler = TopPSampler(temperature: 1.0, topK: 2)
        let counts = sampleCounts(sampler: sampler, logits: log(probs), draws: draws)

        assertOnlySampled(counts, allowedTokens: [0, 3])
        XCTAssertEqual(frequency(counts, token: 0, draws: draws), 0.6667, accuracy: 0.06)
        XCTAssertEqual(frequency(counts, token: 3, draws: draws), 0.3333, accuracy: 0.06)
    }

    func testMinPSamplerKeepsOnlyHighProbabilityToken() {
        let sampler = TopPSampler(temperature: 1.0, minP: 0.95)
        let logits = MLXArray([0.0 as Float, 0.0 as Float, 4.0 as Float])[.newAxis, .ellipsis]

        for _ in 0 ..< 10 {
            let token = sampler.sample(logits: logits).item(Int.self)
            XCTAssertEqual(token, 2)
        }
    }

    func testMinPSamplerLowThresholdKeepsExpectedDistribution() {
        let probs = MLXArray([0.9 as Float, 0.0 as Float, 0.0 as Float, 0.1 as Float])[
            .newAxis, .ellipsis]
        let draws = 4000
        let sampler = TopPSampler(temperature: 1.0, minP: 0.05)
        let counts = sampleCounts(sampler: sampler, logits: log(probs), draws: draws)

        assertOnlySampled(counts, allowedTokens: [0, 3])
        XCTAssertEqual(frequency(counts, token: 0, draws: draws), 0.9, accuracy: 0.05)
        XCTAssertEqual(frequency(counts, token: 3, draws: draws), 0.1, accuracy: 0.05)
    }

    func testGenerateParametersCreatesExpectedSampler() {
        XCTAssertTrue(GenerateParameters(temperature: 0.7, topK: 40).sampler() is TopPSampler)
        XCTAssertTrue(GenerateParameters(temperature: 0.7, minP: 0.1).sampler() is TopPSampler)
        XCTAssertTrue(GenerateParameters(temperature: 0).sampler() is ArgMaxSampler)
    }

    func testSpeculativeSamplingDistributionHonorsTopK() {
        let controller = SpeculativeSamplingController(
            parameters: GenerateParameters(
                temperature: 1.0, topP: 1.0, topK: 2, minP: 0.0, randomSeed: 7))
        let logits =
            MLXArray([0.0 as Float, 4.0 as Float, 3.0 as Float, 1.0 as Float])[.newAxis, .ellipsis]
        let probabilities = controller.probabilities(logits: logits)[0].asArray(Float.self)

        XCTAssertEqual(probabilities[0], 0, accuracy: 1e-6)
        XCTAssertGreaterThan(probabilities[1], 0)
        XCTAssertGreaterThan(probabilities[2], 0)
        XCTAssertEqual(probabilities[3], 0, accuracy: 1e-6)
        XCTAssertEqual(probabilities.reduce(0, +), 1, accuracy: 1e-5)
    }

    func testSpeculativeSamplingAcceptsWhenTargetDominatesDraft() {
        let controller = SpeculativeSamplingController(
            parameters: GenerateParameters(temperature: 1.0, randomSeed: 11))
        let target = MLXArray([0.9 as Float, 0.1 as Float])[.newAxis, .ellipsis]
        let draft = MLXArray([0.4 as Float, 0.6 as Float])[.newAxis, .ellipsis]
        let token = MLXArray([UInt32(0)])

        let decision = controller.acceptOrCorrect(
            draftToken: token,
            targetProbabilities: target,
            draftProbabilities: draft)

        XCTAssertTrue(decision.accepted)
        XCTAssertEqual(decision.acceptanceProbability, 1, accuracy: 1e-6)
        XCTAssertNil(decision.correction)
    }

    func testSpeculativeSamplingRejectsWithResidualCorrection() {
        let controller = SpeculativeSamplingController(
            parameters: GenerateParameters(temperature: 1.0, randomSeed: 13))
        let target = MLXArray([0.0 as Float, 1.0 as Float])[.newAxis, .ellipsis]
        let draft = MLXArray([1.0 as Float, 0.0 as Float])[.newAxis, .ellipsis]
        let token = MLXArray([UInt32(0)])

        let decision = controller.acceptOrCorrect(
            draftToken: token,
            targetProbabilities: target,
            draftProbabilities: draft)

        XCTAssertFalse(decision.accepted)
        XCTAssertEqual(decision.acceptanceProbability, 0, accuracy: 1e-6)
        XCTAssertEqual(decision.correction?.item(Int.self), 1)
    }

    /// `presence_penalty` is defined over GENERATED tokens only.
    ///
    /// The prompt leg is the whole point: before this contract, the ring was
    /// seeded from the prompt, so at the first decode step the window was 100%
    /// prompt and every opening token was penalised for the USER's wording.
    /// vLLM — which defines this parameter in practice, HuggingFace having no
    /// equivalent — feeds `prompt_tokens_tensor` to `repetition_penalties`
    /// alone and computes presence from `output_mask`.
    func testPresencePenaltyContextPenalizesGeneratedTokensNotPrompt() {
        var processor = PresencePenaltyContext(presencePenalty: 0.5, presenceContextSize: 20)
        let logits =
            MLXArray([1.0 as Float, 2.0 as Float, 3.0 as Float, 4.0 as Float])[.newAxis, .ellipsis]

        // A token seen ONLY in the prompt must not be penalised at all.
        processor.prompt(MLXArray([1, 1, 3]))
        let promptOnly = processor.process(logits: logits)[0].asArray(Float.self)
        XCTAssertEqual(promptOnly[1], 2.0, accuracy: 1e-6)
        XCTAssertEqual(promptOnly[3], 4.0, accuracy: 1e-6)

        // Generated tokens are penalised once each, however often they recur —
        // presence is a set membership, not a count.
        processor.didSample(token: MLXArray([1]))
        processor.didSample(token: MLXArray([1]))
        processor.didSample(token: MLXArray([3]))
        let values = processor.process(logits: logits)[0].asArray(Float.self)
        XCTAssertEqual(values[0], 1.0, accuracy: 1e-6)
        XCTAssertEqual(values[1], 1.5, accuracy: 1e-6)
        XCTAssertEqual(values[2], 3.0, accuracy: 1e-6)
        XCTAssertEqual(values[3], 3.5, accuracy: 1e-6)
    }

    /// `frequency_penalty` scales with a COUNT, which is why prompt seeding was
    /// worse here in kind rather than degree: a word the user happened to
    /// repeat three times was suppressed three times as hard.
    func testFrequencyPenaltyContextPenalizesGeneratedCountsNotPrompt() {
        var processor = FrequencyPenaltyContext(frequencyPenalty: 0.5, frequencyContextSize: 20)
        let logits =
            MLXArray([1.0 as Float, 2.0 as Float, 3.0 as Float, 4.0 as Float])[.newAxis, .ellipsis]

        // Repeated PROMPT tokens contribute nothing.
        processor.prompt(MLXArray([1, 1, 3]))
        let promptOnly = processor.process(logits: logits)[0].asArray(Float.self)
        XCTAssertEqual(promptOnly[1], 2.0, accuracy: 1e-6)
        XCTAssertEqual(promptOnly[3], 4.0, accuracy: 1e-6)

        // The same multiset, once GENERATED, is penalised per occurrence:
        // token 1 twice (-1.0), token 3 once (-0.5).
        processor.didSample(token: MLXArray([1]))
        processor.didSample(token: MLXArray([1]))
        processor.didSample(token: MLXArray([3]))
        let values = processor.process(logits: logits)[0].asArray(Float.self)
        XCTAssertEqual(values[0], 1.0, accuracy: 1e-6)
        XCTAssertEqual(values[1], 1.0, accuracy: 1e-6)
        XCTAssertEqual(values[2], 3.0, accuracy: 1e-6)
        XCTAssertEqual(values[3], 3.5, accuracy: 1e-6)
    }

    /// The three processors share a shape but not a specification:
    /// `repetition_penalty` is HuggingFace's, defined over `input_ids` — prompt
    /// INCLUDED — so it must still see the prompt after presence/frequency
    /// stopped doing so. Pinning the divergence keeps a future "consistency"
    /// cleanup from collapsing them back together.
    func testRepetitionPenaltyStillSeesThePromptUnlikeTheOthers() {
        // Each processor gets its OWN logits. `RepetitionContext.process` writes
        // through `logits[0..., indices] = …`, and MLXArray is a reference to
        // shared storage — sharing one array here would feed the second
        // processor the first one's output and quietly compare the wrong thing.
        func freshLogits() -> MLXArray {
            MLXArray([1.0 as Float, 2.0 as Float, 3.0 as Float, 4.0 as Float])[.newAxis, .ellipsis]
        }

        var repetition = RepetitionContext(repetitionPenalty: 1.5, repetitionContextSize: 20)
        repetition.prompt(MLXArray([1, 1, 3]))
        let repeated = repetition.process(logits: freshLogits())[0].asArray(Float.self)
        // HuggingFace's rule: a positive logit is DIVIDED by the penalty.
        XCTAssertEqual(repeated[1], 2.0 / 1.5, accuracy: 1e-6)
        XCTAssertEqual(repeated[3], 4.0 / 1.5, accuracy: 1e-6)
        // Token 2 was never in the prompt, so it is untouched either way.
        XCTAssertEqual(repeated[2], 3.0, accuracy: 1e-6)

        var presence = PresencePenaltyContext(presencePenalty: 0.5, presenceContextSize: 20)
        presence.prompt(MLXArray([1, 1, 3]))
        let unpenalised = presence.process(logits: freshLogits())[0].asArray(Float.self)
        XCTAssertEqual(unpenalised[1], 2.0, accuracy: 1e-6)
        XCTAssertEqual(unpenalised[3], 4.0, accuracy: 1e-6)
    }

    func testGenerateParametersCreatesExpectedPenaltyProcessor() {
        XCTAssertNotNil(GenerateParameters(repetitionPenalty: 1.1).processor())
        XCTAssertNotNil(GenerateParameters(presencePenalty: 0.5).processor())
        XCTAssertNotNil(GenerateParameters(frequencyPenalty: 0.5).processor())
        XCTAssertNotNil(
            GenerateParameters(
                repetitionPenalty: 1.1, presencePenalty: 0.5, frequencyPenalty: 0.5
            ).processor()
        )
    }

    /// 2026-04-30 (Bug 3a): `repetition_penalty: 1.0` is the HuggingFace
    /// idiom for "no penalty" — multiplying / dividing logits by 1.0 is
    /// a no-op. Models like Nemotron-3-Nano-Omni ship `1.0` as a default
    /// in their `generation_config.json`. The processor should treat
    /// this as if no penalty were configured (returning nil), which
    /// also avoids exposing a latent mlx-swift indexing panic on first
    /// decode.
    func testRepetitionPenaltyOneIsTreatedAsNoOp() {
        // The no-op cases — all should produce no processor.
        XCTAssertNil(GenerateParameters(repetitionPenalty: 1.0).processor())
        XCTAssertNil(GenerateParameters(repetitionPenalty: 0.0).processor())
        XCTAssertNil(GenerateParameters(repetitionPenalty: nil).processor())

        // Non-no-op penalties must still produce a processor.
        XCTAssertNotNil(GenerateParameters(repetitionPenalty: 1.05).processor())
        XCTAssertNotNil(GenerateParameters(repetitionPenalty: 0.99).processor())
        XCTAssertNotNil(GenerateParameters(repetitionPenalty: 2.0).processor())

        // Boundary values just outside the no-op equality must still build
        // a processor (we do an exact `!= 1.0` comparison, not approximate).
        XCTAssertNotNil(
            GenerateParameters(repetitionPenalty: Float(1.0).nextUp).processor()
        )
        XCTAssertNotNil(
            GenerateParameters(repetitionPenalty: Float(1.0).nextDown).processor()
        )

        // Combination: 1.0 rep penalty but non-zero presence still produces
        // a processor (driven by presence; rep contribution is correctly nil).
        XCTAssertNotNil(
            GenerateParameters(repetitionPenalty: 1.0, presencePenalty: 0.5).processor()
        )
        // Symmetric: 1.0 rep + non-zero frequency → processor exists.
        XCTAssertNotNil(
            GenerateParameters(repetitionPenalty: 1.0, frequencyPenalty: 0.5).processor()
        )

        // Sanity: presence/frequency keep their additive-no-op semantics
        // (penalty == 0 → no processor); 1.0 is NOT a no-op for them.
        XCTAssertNil(GenerateParameters(presencePenalty: 0.0).processor())
        XCTAssertNotNil(GenerateParameters(presencePenalty: 1.0).processor())
        XCTAssertNil(GenerateParameters(frequencyPenalty: 0.0).processor())
        XCTAssertNotNil(GenerateParameters(frequencyPenalty: 1.0).processor())

        // contextSize=0 also short-circuits regardless of penalty value.
        XCTAssertNil(
            GenerateParameters(repetitionPenalty: 1.5, repetitionContextSize: 0).processor()
        )
    }

    func testPresencePenaltyContextPenalizesUniqueSeenTokens() {
        var processor = PresencePenaltyContext(presencePenalty: 0.5, presenceContextSize: 5)
        for token in [0, 0, 0, 1, 1] {
            processor.didSample(token: MLXArray([token]))
        }

        let logits = MLXArray.zeros([1, 4], type: Float.self)
        let processed = processor.process(logits: logits)
        let values = processed[0].asArray(Float.self)

        XCTAssertEqual(values[0], -0.5, accuracy: 1e-6)
        XCTAssertEqual(values[1], -0.5, accuracy: 1e-6)
        XCTAssertEqual(values[2], 0.0, accuracy: 1e-6)
        XCTAssertEqual(values[3], 0.0, accuracy: 1e-6)
    }

    func testFrequencyPenaltyContextPenalizesByTokenCount() {
        var processor = FrequencyPenaltyContext(frequencyPenalty: 0.5, frequencyContextSize: 5)
        for token in [0, 0, 0, 1, 1] {
            processor.didSample(token: MLXArray([token]))
        }

        let logits = MLXArray.zeros([1, 4], type: Float.self)
        let processed = processor.process(logits: logits)
        let values = processed[0].asArray(Float.self)

        XCTAssertEqual(values[0], -1.5, accuracy: 1e-6)
        XCTAssertEqual(values[1], -1.0, accuracy: 1e-6)
        XCTAssertEqual(values[2], 0.0, accuracy: 1e-6)
        XCTAssertEqual(values[3], 0.0, accuracy: 1e-6)
    }

    func testGenerateParametersPenaltyProcessorComposesPenaltiesInOrder() {
        var processor = GenerateParameters(
            repetitionPenalty: 1.5, repetitionContextSize: 5,
            presencePenalty: 0.5, presenceContextSize: 5,
            frequencyPenalty: 0.25, frequencyContextSize: 5
        ).processor()
        XCTAssertNotNil(processor)

        // Feed the same multiset through BOTH entry points. Repetition takes it
        // from the prompt, presence/frequency from `didSample`, and with a
        // capacity-5 ring the generated tokens overwrite the prompt-seeded ones
        // one-for-one — so all three windows end up holding [0,0,0,1,1] and the
        // composed result is unchanged from when every processor read the
        // prompt. The expected values below are therefore the same numbers as
        // before the split, which is the point: composition order and arithmetic
        // did not change, only WHERE each processor gets its tokens.
        processor?.prompt(MLXArray([0, 0, 0, 1, 1]))
        for token in [0, 0, 0, 1, 1] {
            processor?.didSample(token: MLXArray([token]))
        }
        let logits = MLXArray([1.0 as Float, 0.5 as Float, 0.0 as Float, -0.5 as Float])[
            .newAxis, .ellipsis
        ]
        let processed = processor?.process(logits: logits)
        guard let values = processed?[0].asArray(Float.self) else {
            XCTFail("Expected processed logits")
            return
        }
        XCTAssertEqual(values[0], -0.5833, accuracy: 1e-4)
        XCTAssertEqual(values[1], -0.6667, accuracy: 1e-4)
        XCTAssertEqual(values[2], 0.0, accuracy: 1e-4)
        XCTAssertEqual(values[3], -0.5, accuracy: 1e-4)
    }

    /// A non-positive context size disables the penalty — 0 AND negatives.
    ///
    /// When the size became optional, the guard moved from `> 0` to `!= 0`, and
    /// a negative slipped through it into the context's init, where `if let x,
    /// x > 0` also failed and dropped it into the UNBOUNDED branch. "Off"
    /// silently became the strongest setting available. Nothing in-tree passes
    /// a negative, which is exactly why only a test keeps this honest.
    func testNonPositiveContextSizeDisablesPenalty() {
        // nil is the enabled default (unbounded), not "off".
        XCTAssertNotNil(
            GenerateParameters(presencePenalty: 0.5, presenceContextSize: nil).processor())
        XCTAssertNotNil(
            GenerateParameters(frequencyPenalty: 0.5, frequencyContextSize: nil).processor())

        // A positive size selects the ring — still enabled.
        XCTAssertNotNil(
            GenerateParameters(presencePenalty: 0.5, presenceContextSize: 20).processor())
        XCTAssertNotNil(
            GenerateParameters(frequencyPenalty: 0.5, frequencyContextSize: 20).processor())

        // 0 disables.
        XCTAssertNil(
            GenerateParameters(presencePenalty: 0.5, presenceContextSize: 0).processor())
        XCTAssertNil(
            GenerateParameters(frequencyPenalty: 0.5, frequencyContextSize: 0).processor())

        // Negatives disable too, as they did before the size became optional.
        XCTAssertNil(
            GenerateParameters(presencePenalty: 0.5, presenceContextSize: -1).processor())
        XCTAssertNil(
            GenerateParameters(frequencyPenalty: 0.5, frequencyContextSize: -1).processor())
        XCTAssertNil(
            GenerateParameters(presencePenalty: 0.5, presenceContextSize: Int.min).processor())
    }
}
