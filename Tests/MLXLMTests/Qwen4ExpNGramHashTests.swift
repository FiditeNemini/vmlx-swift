// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXVLM

/// Goldens computed by running the reference helpers from
/// `mlx_vlm/models/qwen4_exp/language.py` (pure integer code) directly:
/// any drift here means the PLE embedding gathers garbage rows.
@Suite("qwen4_exp n-gram hash parity")
struct Qwen4ExpNGramHashTests {
    @Test("splitmix64 matches the reference")
    func splitmix() {
        #expect(Qwen4ExpNGramHash.splitmix64(0) == 16_294_208_416_658_607_535)
        #expect(Qwen4ExpNGramHash.splitmix64(12345) == 2_454_886_589_211_414_944)
    }

    @Test("layer multipliers match, including the ple-layer seed offset")
    func multipliers() {
        #expect(
            Qwen4ExpNGramHash.layerMultipliers(
                unigramVocabSize: 152_064, ngramSize: 4, pleLayerIndex: 0, seed: 42)
                == [27_654_209_115_627, 49_074_646_729_357, 4_153_004_413_633, 4_742_919_160_989])
        #expect(
            Qwen4ExpNGramHash.layerMultipliers(
                unigramVocabSize: 152_064, ngramSize: 4, pleLayerIndex: 2, seed: 42)
                == [17_966_118_997_155, 21_276_791_440_793, 47_673_772_944_719, 38_069_068_554_069])
    }

    @Test("prime search matches")
    func primes() {
        #expect(Qwen4ExpNGramHash.nthPrimeAfter(99_999, count: 1) == 100_003)
        #expect(Qwen4ExpNGramHash.nthPrimeAfter(99_999, count: 5) == 100_057)
    }

    @Test("head vocab layout is consecutive primes with running offsets")
    func layout() {
        let layout = Qwen4ExpNGramHash.headVocabLayout(
            ngramHeads: 3, pleLayerIndex: 0, ngramVocabSizeBase: 100_000)
        #expect(layout.sizes == [100_003, 100_019, 100_043])
        #expect(layout.offsets == [0, 100_003, 200_022])
        #expect(layout.total == 300_065)
    }
}
