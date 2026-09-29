// Copyright © 2026 Osaurus. All rights reserved.

/// Shader-source generation only. One lane owns 16 consecutive coefficients.
/// The supported K%32 contract makes each lane chunk wholly valid or wholly absent.
/// Keep each coefficient FMA in order; packed-word reuse must not reassociate sums.
enum JANGHDecodeQDot {
    static func source(bits: Int, packed: String, rowBase: String, columnBase: String,
                       values: String, accumulator: String, alpha: Float, beta: Float) -> String {
        precondition([2, 3, 4, 6, 8].contains(bits))
        // `packed` may be constant or device storage in an MLX custom kernel.
        // Infer its address space instead of casting it to a device pointer.
        var lines = ["{", "uint qbase = ((\(columnBase)) * \(bits)u) >> 5u;", "auto qwords = \(packed) + \(rowBase) + qbase;"]
        let units = bits == 3 ? 2 : 16 * bits / 32
        for word in 0 ..< units { lines.append("uint qw\(word) = uint(qwords[\(word)]);") }
        if bits == 3 {
            // A16-value chunk starts at bit0 or16 of a UInt32 word and spans48bits.
            // Two words suffice in either case; K%32 keeps the final pair in-row.
            lines.append("uint qstart = ((\(columnBase)) * 3u) & 31u;")
        }
        if bits == 2 {
            // The centered codes are -1.5, -0.5, 0.5, 1.5. Evaluate the
            // same polynomial for the two magnitudes without changing the
            // coefficient FMA sequence or rounding levels on the host.
            lines.append("const float qsmall = 0.5f * fma(\(beta)f, 0.25f, \(alpha)f);")
            lines.append("const float qlarge = 1.5f * fma(\(beta)f, 2.25f, \(alpha)f);")
        }
        for i in 0 ..< 16 {
            let code: String
            if bits == 3 {
                // The branch avoids undefined shifts by32 and preserves small-buffer space.
                lines.append("uint qb\(i) = qstart + \(i * 3)u;")
                lines.append("uint qc\(i) = qb\(i) < 32u ? (qw0 >> qb\(i)) : (qw1 >> (qb\(i) - 32u));")
                lines.append("if (qb\(i) > 29u && qb\(i) < 32u) qc\(i) |= qw1 << (32u - qb\(i));")
                code = "qc\(i)"
            } else {
                let bit = i * bits, word = bit / 32, shift = bit % 32
                var unpacked = "(qw\(word) >> \(shift)u)"
                if shift + bits > 32 {
                    unpacked = "(\(unpacked) | (qw\(word + 1) << \(32 - shift)u))"
                }
                code = unpacked
            }
            if bits == 2 {
                lines.append("{ uint qc = \(code) & 3u; float magnitude = (qc == 0u || qc == 3u) ? qlarge : qsmall; float level = (qc & 2u) != 0u ? magnitude : -magnitude; \(accumulator) = fma(\(values)[\(i)], level, \(accumulator)); }")
            } else {
                lines.append("{ float u = float(\(code) & \((1 << bits) - 1)u) - \(Float((1 << bits) - 1) / 2)f; float level = u * fma(\(beta)f, u * u, \(alpha)f); \(accumulator) = fma(\(values)[\(i)], level, \(accumulator)); }")
            }
        }
        lines.append("}")
        return lines.joined(separator: "\n")
    }
}
