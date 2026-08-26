// Copyright © 2026 Apple Inc.

// Qwen 3.8 Next Flash (qwen4_exp) — hashed n-gram PLE vocabulary machinery.
//
// The n-gram embedding ids are produced by deterministic integer hashing:
// splitmix64-derived odd multipliers per shift position, XOR-mixed shifted
// token ids, reduced modulo per-head PRIME vocab sizes laid out
// consecutively. Every constant must match the reference
// (`mlx_vlm/models/qwen4_exp/language.py`) bit-for-bit or the model gathers
// garbage rows from a 100 GB embedding — goldens are pinned in
// Qwen4ExpNGramHashTests against values computed by running the reference
// helpers directly.

import Foundation

enum Qwen4ExpNGramHash {
    static let splitmixGamma: UInt64 = 0x9E37_79B9_7F4A_7C15
    static let splitmixM1: UInt64 = 0xBF58_476D_1CE4_E5B9
    static let splitmixM2: UInt64 = 0x94D0_49BB_1331_11EB
    static let layerSeedPrime: UInt64 = 10007

    static func splitmix64(_ value: UInt64) -> UInt64 {
        var v = value &+ splitmixGamma
        v = (v ^ (v >> 30)) &* splitmixM1
        v = (v ^ (v >> 27)) &* splitmixM2
        return v ^ (v >> 31)
    }

    /// Odd multipliers per n-gram shift position for one PLE layer.
    static func layerMultipliers(
        unigramVocabSize: Int, ngramSize: Int, pleLayerIndex: Int, seed: Int
    ) -> [UInt64] {
        let maxLong = UInt64(Int64.max)
        let multiplierMax = maxLong / UInt64(max(unigramVocabSize, 1))
        let halfBound = max(1, multiplierMax / 2)
        let baseSeed = UInt64(bitPattern: Int64(seed))
            &+ layerSeedPrime &* UInt64(pleLayerIndex)
        return (0 ..< ngramSize).map { index in
            let value = baseSeed &+ splitmixGamma &* UInt64(index + 1)
            return 2 &* (splitmix64(value) % halfBound) &+ 1
        }
    }

    static func isPrime(_ value: Int) -> Bool {
        if value < 2 { return false }
        if value % 2 == 0 { return value == 2 }
        var divisor = 3
        while divisor * divisor <= value {
            if value % divisor == 0 { return false }
            divisor += 2
        }
        return true
    }

    static func nthPrimeAfter(_ start: Int, count: Int) -> Int {
        var prime = start
        for _ in 0 ..< count {
            prime += 1
            while !isPrime(prime) { prime += 1 }
        }
        return prime
    }

    /// Per-head prime vocab sizes and their running offsets for one PLE layer.
    static func headVocabLayout(
        ngramHeads: Int, pleLayerIndex: Int, ngramVocabSizeBase: Int
    ) -> (sizes: [Int], offsets: [Int], total: Int) {
        var sizes: [Int] = []
        var offsets: [Int] = []
        var total = 0
        for headIdx in 0 ..< ngramHeads {
            let globalHeadIdx = pleLayerIndex * ngramHeads + headIdx
            let size = nthPrimeAfter(ngramVocabSizeBase - 1, count: globalHeadIdx + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        }
        return (sizes, offsets, total)
    }
}
