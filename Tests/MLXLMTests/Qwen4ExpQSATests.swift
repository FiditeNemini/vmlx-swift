// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXVLM

/// Property tests for the QSA block-selection mask; each pins an invariant
/// of the reference recurrence rather than a specific score value.
@Suite("qwen4_exp QSA block selection", .serialized)
struct Qwen4ExpQSATests {

    private func makeMask(
        pastLen: Int, seqLen: Int, keyLen: Int,
        compressRatio: Int = 4, blockTopK: Int = 2, heads: Int = 2, dim: Int = 8
    ) -> MLXArray? {
        MLXRandom.seed(3)
        let numBlocks = keyLen / compressRatio
        let q = MLXRandom.normal([1, heads, seqLen, dim])
        let pooled = MLXRandom.normal([1, 1, numBlocks, dim])
        return Qwen4ExpQSA.selectedTokenMask(
            query: q, pooledKeys: pooled, pastLen: pastLen,
            compressRatio: compressRatio, blockTopK: blockTopK, keyLen: keyLen)
    }

    @Test("below the sparse threshold returns nil (dense fallback)")
    func denseFallback() throws {
        try MLXMetalTestLock.withLock {
            // 8 keys / ratio 4 = 2 complete blocks, not > topk 2.
            #expect(makeMask(pastLen: 4, seqLen: 4, keyLen: 8) == nil)
        }
    }

    @Test("mask is causal: nothing beyond each query position attends")
    func causality() throws {
        try MLXMetalTestLock.withLock {
            let mask = try #require(makeMask(pastLen: 28, seqLen: 4, keyLen: 32))
            for t in 0 ..< 4 {
                let queryEnd = 28 + t + 1
                for j in queryEnd ..< 32 {
                    #expect(
                        mask[0, 0, t, j].item(Bool.self) == false,
                        "t=\(t) attends future token \(j)")
                }
            }
        }
    }

    @Test("the incomplete tail up to the query always attends when sparse")
    func tailInclusion() throws {
        try MLXMetalTestLock.withLock {
            // pastLen 29 → query 0 ends at 30: blocks 0..6 complete (28 tokens),
            // tail 28..29 must be included.
            let mask = try #require(makeMask(pastLen: 29, seqLen: 2, keyLen: 32))
            for t in 0 ..< 2 {
                let queryEnd = 29 + t + 1
                let tailStart = (queryEnd / 4) * 4
                for j in tailStart ..< queryEnd {
                    #expect(
                        mask[0, 0, t, j].item(Bool.self) == true,
                        "t=\(t) tail token \(j) excluded")
                }
            }
        }
    }

    @Test("sparse rows keep exactly blockTopK complete blocks")
    func budget() throws {
        try MLXMetalTestLock.withLock {
            let compressRatio = 4, blockTopK = 2
            let mask = try #require(
                makeMask(
                    pastLen: 31, seqLen: 1, keyLen: 32,
                    compressRatio: compressRatio, blockTopK: blockTopK))
            let queryEnd = 32
            let completeCount = queryEnd / compressRatio  // 8 complete blocks
            var selectedComplete = 0
            for b in 0 ..< completeCount {
                let start = b * compressRatio
                // skip the tail block (== completeCount*ratio.. none here since 32%4==0)
                var allOn = true
                for j in start ..< min(start + compressRatio, queryEnd) {
                    if mask[0, 0, 0, j].item(Bool.self) == false { allOn = false; break }
                }
                if allOn { selectedComplete += 1 }
            }
            // 32 % 4 == 0 → no tail; exactly blockTopK complete blocks attend.
            #expect(selectedComplete == blockTopK, "selected \(selectedComplete)")
        }
    }
}
