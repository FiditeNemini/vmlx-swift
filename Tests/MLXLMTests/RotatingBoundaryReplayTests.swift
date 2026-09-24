import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXLMCommon

/// An exact token recorder: both cache layers contain the token IDs, making
/// a stale or mutated prefix observable after disk restore and continuation.
private final class BoundaryRecordingModel: Module, LanguageModel, @unchecked Sendable {
    var vocabularySize: Int { 64 }
    private(set) var forwardedCount = 0

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple(), RotatingKVCache(maxSize: 16, keep: 0, step: 16)]
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        let step = max(1, windowSize ?? 512)
        var tokens = input.text.tokens.reshaped(-1)
        while tokens.size > step {
            _ = callAsFunction(tokens[..<step][.newAxis], cache: cache)
            MLX.eval(cache)
            tokens = tokens[step...]
        }
        return .tokens(LMInput.Text(tokens: tokens))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let count = inputs.size
        forwardedCount += count
        let keys = inputs.asType(.float32).reshaped(1, 1, count, 1)
        for layer in cache ?? [] {
            _ = layer.update(keys: keys, values: keys * 2)
        }
        return MLXArray.zeros([1, count, vocabularySize])
    }
}

@Suite("Rotating boundary replay", .serialized)
struct RotatingBoundaryReplayTests {
    @Test(arguments: [39, 35, 15])
    func persistedBoundariesMatchIndependentPrefill(length: Int) async throws {
        try await verify(length: length, masked: false)
    }

    @Test
    func maskedInputKeepsIndependentReplay() async throws {
        try await verify(length: 39, masked: true)
    }

    private func verify(length: Int, masked: Bool) async throws {
        try await MLXMetalTestLock.withLock {
            let root = FileManager.default.temporaryDirectory
                .appendingPathComponent("rotating-replay-test-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: root) }
            let model = BoundaryRecordingModel()
            var configuration = ModelConfiguration(id: "rotating-replay-test")
            // Synthetic fixture only: finish after prefill so recorded work is
            // exactly the prompt plus boundary reconstruction, never sampling.
            configuration.eosTokenIds = Set(0..<64)
            let processor = TestInputProcessor(
                tokenizer: TestTokenizer(vocabularySize: 64),
                configuration: configuration,
                messageGenerator: DefaultMessageGenerator())
            nonisolated(unsafe) let context = ModelContext(
                configuration: configuration, model: model,
                processor: processor, tokenizer: processor.tokenizer)
            let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
                usePagedCache: false, enableDiskCache: true,
                diskCacheMaxGB: 0.1, diskCacheDir: root,
                modelKey: "rotating-replay-test"))
            let engine = BatchEngine(context: context, maxBatchSize: 1, cacheCoordinator: coordinator)
            let ids = Array(1...length)
            let boundary = length - 3
            let input = LMInput(
                text: LMInput.Text(
                    tokens: MLXArray(ids.map(Int32.init)),
                    mask: masked ? MLXArray.ones([length], dtype: .bool) : nil),
                cachePrefixTokenCounts: [boundary],
                cacheStablePrefixTokenCounts: [boundary])
            let parameters = GenerateParameters(maxTokens: 1, temperature: 0, prefillStepSize: 16)
            let salt = computeCacheSalt(for: input, parameters: parameters)
            let (_, stream) = await engine.submit(input: input, parameters: parameters)
            var stop: GenerateStopReason?
            for await event in stream {
                if case .info(let info) = event { stop = info.stopReason }
            }
            await engine.shutdown()
            #expect(stop == .stop)
            // At 39, all three wrapped boundaries share the completed 32-token
            // chunk prefix. At 35 the earlier boundary crosses that chunk edge.
            if length == 39 { #expect(model.forwardedCount == (masked ? 148 : 84)) }
            if length == 35 { #expect(model.forwardedCount == 116) }
            if length == 15 { #expect(model.forwardedCount == 15) }

            for count in [length - 1, boundary - 1, boundary] {
                let prefix = Array(ids.prefix(count))
                let arrays = try #require(coordinator.diskCache?.fetch(tokens: prefix, mediaSalt: salt))
                var restored = model.newCache(parameters: parameters)
                #expect(restoreFromDiskArrays(arrays, into: &restored) == count)
                let reference = model.newCache(parameters: parameters)
                let result = try model.prepare(
                    LMInput(tokens: MLXArray(prefix.map(Int32.init))),
                    cache: reference, windowSize: 16)
                if case .tokens(let remaining) = result {
                    _ = model(remaining[text: .newAxis], cache: reference, state: nil)
                }
                MLX.eval(reference, restored)
                for (expected, actual) in zip(reference, restored) {
                    #expect(actual.offset == expected.offset)
                    #expect(actual.metaState == expected.metaState)
                    #expect(actual.state.count == expected.state.count)
                    for (a, b) in zip(expected.state, actual.state) {
                        #expect(a.shape == b.shape)
                        #expect(a.asType(.float32).asArray(Float.self) == b.asType(.float32).asArray(Float.self))
                    }
                }
                // Advancing one restored boundary must not mutate another seed
                // or change the next cache state compared with a cold prefix.
                let suffix = MLXArray([Int32(7), Int32(9)])[.newAxis]
                _ = model(suffix, cache: reference)
                _ = model(suffix, cache: restored)
                MLX.eval(reference, restored)
                for (a, b) in zip(reference, restored) {
                    #expect(a.metaState == b.metaState)
                    for (x, y) in zip(a.state, b.state) {
                        #expect(x.asType(.float32).asArray(Float.self) == y.asType(.float32).asArray(Float.self))
                    }
                }
            }
        }
    }
}
