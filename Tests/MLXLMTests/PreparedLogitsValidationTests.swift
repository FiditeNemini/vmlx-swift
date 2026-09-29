import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

private final class PreparedLogitsFixture: Module, LanguageModel, @unchecked Sendable {
    var outputShape: [Int]
    var preparedOffsets: [Int] = []
    var preparedTokenCounts: [Int] = []
    let failProjection: Bool
    let mutateCache: Bool
    var vocabularySize: Int { 4 }

    init(shape: [Int], failProjection: Bool = false, mutateCache: Bool = false) {
        self.outputShape = shape
        self.failProjection = failProjection
        self.mutateCache = mutateCache
        super.init()
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        mutateCache ? [KVCacheSimple()] : []
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        preparedOffsets.append(cache.first?.offset ?? 0)
        preparedTokenCounts.append(input.text.tokens.size)
        if mutateCache {
            let row = MLXArray.ones([1, 1, input.text.tokens.size, 4])
            for layer in cache { _ = layer.update(keys: row, values: row) }
        }
        if failProjection {
            let invalid = matmul(MLXArray.zeros([2, 3]), MLXArray.zeros([4, 2]))
            return .logits(LMOutput(logits: invalid))
        }
        let count = outputShape.reduce(1, *)
        let logits = MLXArray(0 ..< count).asType(.float32).reshaped(outputShape)
        return .logits(LMOutput(logits: logits))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray([Float(0), 1, 2, 3]).reshaped(1, 1, 4)
    }
}

final class PreparedLogitsValidationTests: XCTestCase {
    func testSessionInvalidatesFailedPreparationUntilExplicitReset() async throws {
        try await MLXMetalTestLock.withLock {
            let fixture = PreparedLogitsFixture(shape: [1, 1, 4], mutateCache: true)
            let tokenizer = PreparationHistoryTokenizer()
            let processor = TestInputProcessor(
                tokenizer: tokenizer,
                configuration: ModelConfiguration(id: "fixture/session"),
                messageGenerator: DefaultMessageGenerator())
            let context = ModelContext(
                configuration: processor.configuration, model: fixture,
                processor: processor, tokenizer: tokenizer)
            let parameters = GenerateParameters(maxTokens: 1, temperature: 0)
            let session = ChatSession(context, generateParameters: parameters)
            _ = try await session.respond(to: "first fact")
            fixture.outputShape = [1, 4]
            do {
                _ = try await session.respond(to: "failing continuation")
                XCTFail("Expected malformed prepared logits")
            } catch is PreparedLogitsValidationError {}
            XCTAssertGreaterThan(
                fixture.preparedOffsets.last ?? 0, 0,
                "Failure fixture must begin with retained conversation state")
            let callsAfterFailure = fixture.preparedOffsets.count
            await session.withCache { XCTAssertNil($0) }
            fixture.outputShape = [1, 1, 4]
            do {
                _ = try await session.respond(to: "must not silently lose history")
                XCTFail("Invalid session must require explicit reset/history replay")
            } catch ChatSessionError.cacheInvalidatedAfterPreparation {}
            XCTAssertEqual(fixture.preparedOffsets.count, callsAfterFailure)
            let url = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: url) }
            do {
                try await session.saveCache(to: url)
                XCTFail("Failed session cache must not be saved")
            } catch ChatSessionError.cacheInvalidatedAfterPreparation {}
            XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
            await session.clear()
            _ = try await session.respond(to: "explicitly new conversation")
            XCTAssertEqual(fixture.preparedOffsets.last, 0)
            let restored = ChatSession(
                context,
                history: [.user("first fact"), .assistant("remembered fact")],
                generateParameters: parameters)
            _ = try await restored.respond(to: "continuation after replay")
            XCTAssertEqual(fixture.preparedOffsets.last, 0)
            XCTAssertEqual(
                fixture.preparedTokenCounts.last, 3,
                "A replacement session must prepare complete supplied history plus next turn")
        }
    }

    func testMalformedPreparedShapesThrowBeforeSampling() throws {
        try MLXMetalTestLock.withLock {
            for shape in [[], [4], [1, 4], [1, 1, 1, 4], [0, 1, 4], [1, 0, 4], [1, 1, 0]] {
                XCTAssertThrowsError(
                    try TokenIterator(
                        input: LMInput(tokens: MLXArray([Int32(1)])),
                        model: PreparedLogitsFixture(shape: shape),
                        parameters: GenerateParameters(maxTokens: 1, temperature: 0))
                ) { error in
                    XCTAssertEqual(error as? PreparedLogitsValidationError, .invalidShape(shape))
                }
            }
        }
    }

    func testOriginalProjectionErrorIsNotReplacedByShapeError() throws {
        try MLXMetalTestLock.withLock {
            XCTAssertThrowsError(
                try TokenIterator(
                    input: LMInput(tokens: MLXArray([Int32(1)])),
                    model: PreparedLogitsFixture(shape: [1, 1, 4], failProjection: true),
                    parameters: GenerateParameters(maxTokens: 1, temperature: 0))
            ) { error in
                guard let mlxError = error as? MLXError, case .caught(let message) = mlxError else {
                    return XCTFail("Expected originating MLX projection error, got \(error)")
                }
                XCTAssertTrue(message.contains("matmul"), message)
                XCTAssertFalse(error is PreparedLogitsValidationError)
            }
            // A fresh valid request must not inherit the failed request's scope.
            var iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray([Int32(1)])),
                model: PreparedLogitsFixture(shape: [1, 1, 4]),
                parameters: GenerateParameters(maxTokens: 1, temperature: 0))
            XCTAssertEqual(iterator.next(), 3)
        }
    }

    func testFailedPrepareDoesNotPublishMutatedCache() throws {
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("prepared-logits-failure-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: directory) }
            let coordinator = CacheCoordinator(
                config: CacheCoordinatorConfig(
                    usePagedCache: false, enableDiskCache: true, diskCacheDir: directory,
                    modelKey: "prepared-logits-failure"))
            let input = LMInput(tokens: MLXArray([Int32(1)]))
            let parameters = GenerateParameters(maxTokens: 1, temperature: 0)
            let callerCache = KVCacheSimple()
            XCTAssertThrowsError(
                try TokenIterator(
                    input: input, model: PreparedLogitsFixture(shape: [1, 4], mutateCache: true),
                    cache: [callerCache], parameters: parameters, cacheCoordinator: coordinator))
            // A throwing prepare is not transactional for caller-owned objects.
            // Establish that this fixture really mutated one before the failure.
            XCTAssertEqual(callerCache.offset, 1)
            XCTAssertEqual(coordinator.diskCache?.stores, 0)
            let salt = computeCacheSalt(for: input, parameters: parameters)
            guard case .miss = coordinator.fetch(tokens: [1], mediaSalt: salt) else {
                return XCTFail("Failed prepared output must not publish a prefix checkpoint")
            }
            // Discard the caller cache after failure; test the supported fresh
            // request path without implicitly claiming rollback or B=2 support.
            var next = try TokenIterator(
                input: input, model: PreparedLogitsFixture(shape: [1, 1, 4]),
                parameters: parameters, cacheCoordinator: coordinator)
            XCTAssertEqual(next.next(), 3)
            XCTAssertEqual(coordinator.diskCache?.stores, 0)
        }
    }

    func testValidSingleAndBatchedGeometryPreservesLastRows() throws {
        try MLXMetalTestLock.withLock {
            for shape in [[1, 1, 4], [1, 3, 4], [2, 1, 4], [2, 3, 4]] {
                let logits = MLXArray(0 ..< shape.reduce(1, *)).asType(.float32).reshaped(shape)
                try validatePreparedLogitsForSampling(logits)
                let row = logits[0..., -1, 0...]
                let expected = (0 ..< shape[0]).flatMap { batch in
                    (0 ..< shape[2]).map {
                        Float((batch * shape[1] + shape[1] - 1) * shape[2] + $0)
                    }
                }
                XCTAssertEqual(row.asArray(Float.self), expected)
            }
            // TokenIterator is single-request; batched geometry above tests only
            // the rank contract, not scalar iteration over multiple sequences.
            for length in [1, 3] {
                var iterator = try TokenIterator(
                    input: LMInput(tokens: MLXArray([Int32(1)])),
                    model: PreparedLogitsFixture(shape: [1, length, 4]),
                    parameters: GenerateParameters(maxTokens: 1, temperature: 0))
                XCTAssertEqual(iterator.next(), 3)
            }
        }
    }
}

private struct PreparationHistoryTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "answer" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { "answer" }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        Array(repeating: 1, count: messages.count)
    }
}
