import MLX
import MLXFast
import XCTest
@testable import MLXLMCommon

final class JANGHDecodeQDotTests: XCTestCase {
    func testTwoBitLevelsPreservePolynomialDotBits() throws {
        try MLXMetalTestLock.withLock {
            let count = 4096
            var words: [UInt32] = []
            var values: [Float] = []
            for row in 0..<count {
                words.append(row < 4 ? UInt32(row) &* 0x55555555 : UInt32(row) &* 2654435761)
                for i in 0..<16 {
                    let v = Float((row * 13 + i * 7) % 31 - 15) / 8
                    values.append(i % 3 == 0 ? -v : v)
                }
            }
            let x = MLXArray(values, [count, 16]), packed = MLXArray(words)
            eval(x, packed)
            // Actual GLM/Naive coefficients, plus zero, sign and cancellation cases.
            var books: [(Float, Float)] = [(0.893, 0.05065)]
            for alpha in [Float(0), 1, -1, 1.3125, 0.1171875] {
                for beta in [Float(0), 0.01, -0.125, 0.375, -1] {
                    books.append((alpha, beta))
                }
            }
            let reference = MLXFast.metalKernel(
                name: "jangh_two_bit_polynomial_reference",
                inputNames: ["x", "packed", "coeff"], outputNames: ["out"],
                source: """
                    uint row = thread_position_in_grid.x;
                    if (row >= 4096u) return;
                    float accum = 0.0f;
                    for (uint i = 0; i < 16; ++i) {
                        uint code = (packed[row] >> (i * 2u)) & 3u;
                        float u = float(code) - 1.5f;
                        float level = u * fma(coeff[1], u * u, coeff[0]);
                        accum = fma(x[row * 16u + i], level, accum);
                    }
                    out[row] = accum;
                    """)
            for (index, book) in books.enumerated() {
                let dot = JANGHDecodeQDot.source(
                    bits: 2, packed: "packed", rowBase: "row", columnBase: "0u",
                    values: "values", accumulator: "accum", alpha: book.0, beta: book.1)
                let candidate = MLXFast.metalKernel(
                    name: "jangh_two_bit_level_candidate_\(index)",
                    inputNames: ["x", "packed"], outputNames: ["out"],
                    source: """
                        uint row = thread_position_in_grid.x;
                        if (row >= 4096u) return;
                        float values[16];
                        for (uint i = 0; i < 16; ++i) values[i] = x[row * 16u + i];
                        float accum = 0.0f;
                        \(dot)
                        out[row] = accum;
                        """)
                let expected = reference(
                    [x, packed, MLXArray([book.0, book.1])],
                    grid: (count, 1, 1), threadGroup: (256, 1, 1),
                    outputShapes: [[count]], outputDTypes: [.float32])[0]
                let actual = candidate(
                    [x, packed], grid: (count, 1, 1), threadGroup: (256, 1, 1),
                    outputShapes: [[count]], outputDTypes: [.float32])[0]
                eval(expected, actual)
                let mismatches = zip(expected.asArray(Float.self), actual.asArray(Float.self))
                    .filter { $0.bitPattern != $1.bitPattern }.count
                XCTAssertEqual(mismatches, 0, "codebook \(index): alpha=\(book.0), beta=\(book.1)")
            }
        }
    }
}
