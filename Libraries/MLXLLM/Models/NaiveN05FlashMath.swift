// Naive-N0.5 reference math. Vendor revision 0235b3b5; no model registration.
import Foundation
import MLX

/// Deliberately explicit reference operations. Optimized paths must match these
/// before adopting fused SDPA (asymmetric values, sinks and all-masked rows).
enum NaiveN05FlashMath {
    static func roundIndexerFP8(_ input: MLXArray) -> MLXArray {
        let x = input.asType(.float32)
        let scale = maximum(abs(x).max(axis: -1, keepDims: true), 1e-4) / 448
        let normalized = clip(x / scale, min: -448, max: 448)
        // E4M3FN subnormal spacing is 2^-9; normals have three mantissa bits.
        let exponent = maximum(floor(log2(maximum(abs(normalized), 1.0 / 512))), -6)
        let spacing = pow(MLXArray(Float(2)), exponent - 3)
        return round(normalized / spacing) * spacing * scale
    }

    /// GPTNeoX half-rotation on only the prefix; positions may differ per batch.
    static func rotary(_ x: MLXArray, positions: MLXArray, dimensions: Int, theta: Double) -> MLXArray {
        precondition(dimensions > 0 && dimensions.isMultiple(of: 2) && dimensions <= x.dim(-1))
        let inverse = exp(-MLXArray(0 ..< dimensions / 2).asType(.float32)
            * (Float(2 * log(theta)) / Float(dimensions)))
        let angle = positions.asType(.float32).expandedDimensions(axis: -1) * inverse
        let c = cos(angle).expandedDimensions(axis: 1).asType(x.dtype)
        let s = sin(angle).expandedDimensions(axis: 1).asType(x.dtype)
        let a = x[.ellipsis, ..<(dimensions / 2)]
        let b = x[.ellipsis, (dimensions / 2)..<dimensions]
        let rotated = concatenated([a * c - b * s, b * c + a * s], axis: -1)
        return dimensions == x.dim(-1) ? rotated : concatenated([rotated, x[.ellipsis, dimensions...]], axis: -1)
    }

    static func allowedMask(padding: MLXArray, queryOffset: Int, length: Int, keyOffset: Int, keyLength: Int, window: Int?) -> MLXArray {
        let q = MLXArray(queryOffset ..< queryOffset + length).expandedDimensions(axis: -1)
        let k = MLXArray(keyOffset ..< keyOffset + keyLength)
        var mask = q .>= k
        if let window { mask = mask .&& (q - k .< window) }
        return mask.expandedDimensions(axis: 0) .&& padding[0..., keyOffset ..< keyOffset + keyLength].expandedDimensions(axis: 1)
    }

    /// MLX merge-sort preserves left input on equality. Descending order is
    /// obtained by negating scores rather than reversing sorted equal keys.
    static func sparseMask(scores: MLXArray, allowed: MLXArray, topK: Int) -> MLXArray {
        let masked = which(allowed, scores, MLXArray(-Float.infinity))
        let selected = argSort(-masked, axis: -1)[.ellipsis, ..<min(topK, scores.dim(-1))]
        let picked = putAlong(MLXArray.zeros(scores.shape, dtype: .bool), selected,
            values: MLXArray(true), axis: -1)
        return allowed .&& picked
    }

    static func attention(query: MLXArray, key: MLXArray, value: MLXArray, allowed: MLXArray, sink: MLXArray?, valueScale: Float?) -> MLXArray {
        let repeats = query.dim(1) / key.dim(1)
        let k = repeated(key, count: repeats, axis: 1)
        var v = repeated(value, count: repeats, axis: 1)
        if let valueScale { v = v * MLXArray(valueScale, dtype: v.dtype) }
        var logits = matmul(query, k.swappedAxes(-1, -2)) * Float(1 / sqrt(Double(query.dim(-1))))
        logits = which(allowed.expandedDimensions(axis: 1), logits, MLXArray(-Float.infinity, dtype: logits.dtype))
        if let sink {
            let column = broadcast(sink.asType(logits.dtype).reshaped(1, -1, 1, 1), to: [query.dim(0), query.dim(1), query.dim(2), 1])
            logits = concatenated([logits, column], axis: -1)
        }
        let probabilities = nanToNum(softmax(logits.asType(.float32), axis: -1), nan: 0)[.ellipsis, ..<key.dim(2)].asType(v.dtype)
        return matmul(probabilities, v)
    }
}
