// Copyright © 2026 Osaurus. All rights reserved.

/// Shader-source generation only. One lane owns 16 consecutive coefficients.
/// The supported K%32 contract makes each lane chunk wholly valid or wholly absent.
/// Keep each coefficient FMA in order; packed-word reuse must not reassociate sums.
enum JANGHDecodeQDot {
    static func source(bits: Int, packed: String, rowBase: String, columnBase: String,
                       values: String, accumulator: String, alpha: Float, beta: Float) -> String {
        precondition([2, 3, 4, 6, 8].contains(bits))
        let unitBits = bits == 3 ? 16 : 32
        let unitType = bits == 3 ? "ushort" : "uint"
        let units = 16 * bits / unitBits
        var lines = ["{", "const device \(unitType)* qwords = reinterpret_cast<const device \(unitType)*>(\(packed) + \(rowBase)) + ((\(columnBase)) * \(bits)u / \(unitBits)u);"]
        for word in 0 ..< units { lines.append("uint qw\(word) = uint(qwords[\(word)]);") }
        for i in 0 ..< 16 {
            let bit = i * bits, word = bit / unitBits, shift = bit % unitBits
            var code = "(qw\(word) >> \(shift)u)"
            if shift + bits > unitBits {
                code = "(\(code) | (qw\(word + 1) << \(unitBits - shift)u))"
            }
            lines.append("{ float u = float(\(code) & \((1 << bits) - 1)u) - \(Float((1 << bits) - 1) / 2)f; float level = u * fma(\(beta)f, u * u, \(alpha)f); \(accumulator) = fma(\(values)[\(i)], level, \(accumulator)); }")
        }
        lines.append("}")
        return lines.joined(separator: "\n")
    }
}
