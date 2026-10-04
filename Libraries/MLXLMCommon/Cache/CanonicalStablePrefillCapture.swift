import Foundation
import MLX

/// Model-owned checkpoint signal. Unlike UI progress, this reports a materialized
/// complete chunk from an unchanged, cold Qwen4Exp text preparation.
public enum CanonicalTextPrefillCheckpointReporter {
    private final class Scope {
        let capture: CanonicalStablePrefillCapture
        init(_ capture: CanonicalStablePrefillCapture) { self.capture = capture }
    }
    private static let key = "ai.osaurus.vmlx.canonicalTextPrefillCheckpoint"

    public static var isActive: Bool { Thread.current.threadDictionary[key] is Scope }

    static func withCapture<T>(_ capture: CanonicalStablePrefillCapture?,
                               operation: () throws -> T) rethrows -> T {
        guard let capture else { return try operation() }
        let dictionary = Thread.current.threadDictionary
        let previous = dictionary[key]
        dictionary[key] = Scope(capture)
        defer {
            if let previous { dictionary[key] = previous }
            else { dictionary.removeObject(forKey: key) }
        }
        return try operation()
    }

    public static func reportQwen4ExpColdTextChunk(
        input: LMInput, cache: [KVCache], chunkSize: Int, completed: Int,
        beganWithEmptyCache: Bool
    ) {
        guard let scope = Thread.current.threadDictionary[key] as? Scope else { return }
        scope.capture.receive(input: input, cache: cache, chunkSize: chunkSize,
                              completed: completed, beganWithEmptyCache: beganWithEmptyCache)
    }
}

/// Request-local, one-snapshot retention. Never interprets aligned warm offsets
/// as canonical provenance and never changes the live prepare partition.
final class CanonicalStablePrefillCapture {
    let chunkSize: Int
    let seedCount: Int
    private let promptTokens: [Int]
    private let salt: String?
    private let owners: [ObjectIdentifier]
    private let schema: [String]
    private(set) var snapshot: [KVCache]?
    private(set) var snapshotSeconds: TimeInterval = 0

    init?(input: LMInput, promptTokens: [Int], cache: [KVCache],
          chunkSize: Int, targets: [Int], salt: String?) {
        guard chunkSize > 0, !input.hasMediaContent, !input.requiresPostPrepareCacheKey,
              input.cachePromptIntent != .auxiliary,
              input.text.mask == nil || input.text.mask?.size == promptTokens.count,
              !cache.isEmpty, cache.allSatisfy({ $0.offset == 0 && $0.state.isEmpty }),
              cache.allSatisfy({ type(of: $0) == MambaCache.self || type(of: $0) == QSAKVCache.self }),
              let first = targets.filter({ $0 > 0 && $0 < promptTokens.count }).min()
        else { return nil }
        let seed = (first / chunkSize) * chunkSize
        guard seed > 0 else { return nil }
        self.chunkSize = chunkSize; seedCount = seed
        self.promptTokens = promptTokens; self.salt = salt
        owners = cache.map { ObjectIdentifier($0 as AnyObject) }
        schema = cache.map { String(reflecting: type(of: $0)) }
    }

    func receive(input: LMInput, cache: [KVCache], chunkSize: Int, completed: Int,
                 beganWithEmptyCache: Bool) {
        guard snapshot == nil, !Task.isCancelled, beganWithEmptyCache,
              chunkSize == self.chunkSize, completed == seedCount,
              !input.hasMediaContent,
              let ids = input.text.tokenIds, completed < ids.count,
              input.text.mask == nil || input.text.mask?.size == ids.count,
              ids.count <= promptTokens.count, promptTokens.starts(with: ids),
              cache.map({ ObjectIdentifier($0 as AnyObject) }) == owners,
              cache.map({ String(reflecting: type(of: $0)) }) == schema,
              cache.allSatisfy({ $0.offset == completed }) else { return }
        let start = Date.timeIntervalSinceReferenceDate
        let owned = makePromptBoundaryCacheSnapshot(from: cache)
        guard !Task.isCancelled else { return }
        snapshotSeconds = Date.timeIntervalSinceReferenceDate - start
        snapshot = owned
    }

    func copySeed(for tokens: [Int], salt: String?, chunkSize: Int) -> [KVCache]? {
        guard !Task.isCancelled, salt == self.salt, chunkSize == self.chunkSize,
              tokens.count >= seedCount, tokens.count < promptTokens.count,
              promptTokens.starts(with: tokens), let snapshot,
              snapshot.allSatisfy({ $0.offset == seedCount }),
              snapshot.map({ String(reflecting: type(of: $0)) }) == schema else { return nil }
        return makePromptBoundaryCacheSnapshot(from: snapshot)
    }
}
