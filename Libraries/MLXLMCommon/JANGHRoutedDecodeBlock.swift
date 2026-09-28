import MLX

/// Experimental composition over owned mmap banks, not a SwitchGLU replacement.
/// Callers explicitly supply the vendor activation limit and result dtype.
/// No factory or prefill dispatch uses this block.
final class JANGHRoutedDecodeBlock {
    private let gate: JANGHMappedBanks.Projection
    private let up: JANGHMappedBanks.Projection
    private let down: JANGHMappedBanks.Projection
    private let gateUpKernel: JANGHFusedGateUpKernel
    private let downKernel: JANGHWeightedDownKernel
    private let limit: Float?

    init(banks: JANGHMappedBanks, parentModule: String, activationLimit: Float?) throws {
        if let activationLimit, !activationLimit.isFinite || activationLimit <= 0 {
            throw JANGHFormatContract.ValidationError.invalid("invalid JANGH activation limit")
        }
        gate = try banks.projection(parentModule + ".gate_proj")
        up = try banks.projection(parentModule + ".up_proj")
        down = try banks.projection(parentModule + ".down_proj")
        downKernel = try JANGHWeightedDownKernel(contract: banks.contract, module: parentModule + ".down_proj")
        gateUpKernel = try JANGHFusedGateUpKernel(
            contract: banks.contract, gateModule: parentModule + ".gate_proj",
            upModule: parentModule + ".up_proj", outputRotation: downKernel.inputRotation)
        limit = activationLimit
    }

    func callAsFunction(
        _ input: MLXArray, indices: MLXArray, scores: MLXArray, outputDType: DType
    ) throws -> MLXArray {
        guard input.ndim == 2, indices.ndim == 2, indices.dim(0) == input.dim(0),
            scores.shape == indices.shape, scores.dtype == .float32,
            [.float16, .bfloat16, .float32].contains(outputDType)
        else { throw JANGHFormatContract.ValidationError.invalid("invalid JANGH decode routing") }
        let prepared = try gateUpKernel.prepareInputForFusedDecode(input)
        let hidden = try gateUpKernel.activatePreparedInput(
            prepared, gatePacked: gate.packed, gateScales: gate.scales,
            upPacked: up.packed, upScales: up.scales, indices: indices, limit: limit)
        return try downKernel.projectPreparedHidden(
            hidden, preparedBasis: downKernel.inputRotation,
            packed: down.packed, scales: down.scales,
            indices: indices, scores: scores, outputDType: outputDType)
    }
}
