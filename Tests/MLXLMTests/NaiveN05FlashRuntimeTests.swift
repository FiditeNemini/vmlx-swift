import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import XCTest
@testable import MLXLLM

final class NaiveN05FlashRuntimeTests: XCTestCase {
    private func tiny() throws -> NaiveN05ArchitectureContract {
        let values: [String: Any] = [
            "model_type": "naive_n05_flash", "hidden_size": 8, "intermediate_size": 12,
            "num_hidden_layers": 2, "vocab_size": 16, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 4, "v_head_dim": 2,
            "swa_num_attention_heads": 2, "swa_num_key_value_heads": 1,
            "swa_head_dim": 4, "swa_v_head_dim": 2, "partial_rotary_factor": 0.5,
            "n_routed_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 6,
            "hybrid_layer_pattern": [0, 1], "moe_layer_freq": [0, 1],
            "index_n_heads": 2, "index_head_dim": 4, "index_top_k": 3,
            "sliding_window": 3,
        ]
        return try JSONDecoder().decode(NaiveN05ArchitectureContract.self,
            from: JSONSerialization.data(withJSONObject: values))
    }
    private func close(_ actual: MLXArray, _ expected: MLXArray, tolerance: Float = 1e-5,
        file: StaticString = #filePath, line: UInt = #line)
    {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertLessThanOrEqual(abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self), tolerance, file: file, line: line)
    }

    func testFP8FullRowScalingAndSubnormalTieEven() {
        MLXMetalTestLock.withLock {
        // Scale=1: includes saturation bin, normal ties (even), subnormal ties.
        let values: [Float] = [448, 1.0625, 1.1875, -1.0625, 1.0 / 1024, 3.0 / 1024, 0]
        let x = MLXArray(values).reshaped(1, 1, 1, -1)
        close(NaiveN05FlashMath.roundIndexerFP8(x), MLXArray([Float(448), 1, 1.25, -1, 0, 1.0 / 256, 0]).reshaped(x.shape), tolerance: 0)
        // Clamp belongs before /448, and scale spans the entire row (not64 groups).
        let tiny = MLXArray([Float(1e-6), -1e-6, 0, 1e-4]).reshaped(1, 1, 1, 4)
        let scaled = NaiveN05FlashMath.roundIndexerFP8(tiny)
        close(scaled, MLXArray([Float(4.5), -4.5, 0, 448]).reshaped(tiny.shape) * Float(1e-4 / 448), tolerance: 1e-10)
        let wide = concatenated([MLXArray.ones([1,1,1,64]) * 1.0625, MLXArray.ones([1,1,1,64]) * 448], axis: -1)
        close(NaiveN05FlashMath.roundIndexerFP8(wide)[.ellipsis, ..<64], MLXArray.ones([1,1,1,64]), tolerance: 0)
            }
    }

    func testPartialRotaryPerBatchPositionsKeepsSuffix() {
        MLXMetalTestLock.withLock {
        let x = MLXArray([Float(1), 0, 7, 9, 1, 0, 7, 9]).reshaped(2, 1, 1, 4)
        let y = NaiveN05FlashMath.rotary(x, positions: MLXArray([0,1]).reshaped(2,1), dimensions: 2, theta: 10_000)
        close(y, MLXArray([Float(1),0,7,9,Float(cos(1.0)),Float(sin(1.0)),7,9]).reshaped(x.shape))
            }
    }

    func testStableSparseTiesAcrossLargeSortAndPadding() {
        MLXMetalTestLock.withLock {
        let scores = MLXArray.zeros([1,1,2053])
        let allowed = MLXArray.ones([1,1,2053], dtype: .bool)
        allowed[0,0,0] = MLXArray(false)
        let mask = NaiveN05FlashMath.sparseMask(scores: scores, allowed: allowed, topK: 2048)
        let actual = mask.asArray(Bool.self)
        XCTAssertFalse(actual[0])
        XCTAssertTrue(actual[1...2048].allSatisfy { $0 })
        XCTAssertTrue(actual[2049...].allSatisfy { !$0 })
            }
    }

    func testSinkAndAllMaskedAsymmetricAttention() {
        MLXMetalTestLock.withLock {
        let q = MLXArray.zeros([1,2,1,4]), k = MLXArray.zeros([1,1,2,4])
        let v = MLXArray([Float(2),4,6,8]).reshaped(1,1,2,2)
        let allowed = MLXArray.ones([1,1,2], dtype: .bool)
        // Two equal real logits plus one equal sink: sum(values)/3, scaledonce.
        let y = NaiveN05FlashMath.attention(query:q,key:k,value:v,allowed:allowed,sink:MLXArray.zeros([2]),valueScale:0.5)
        close(y, MLXArray([Float(4.0/3),2,4.0/3,2]).reshaped(1,2,1,2))
        for sink in [nil, MLXArray([Float(100), -100])] as [MLXArray?] {
            close(NaiveN05FlashMath.attention(query:q,key:k,value:v,allowed:MLXArray.zeros([1,1,2],dtype:.bool),sink:sink,valueScale:nil), MLXArray.zeros([1,2,1,2]), tolerance:0)
        }
            }
    }

    func testSlidingWindowReturnsCallHistoryAndRejectsRollbackAfterWrap() throws {
        try MLXMetalTestLock.withLock {
        let cache = NaiveN05FlashCache(window:3,requiresIndexer:false)
        let first = MLXArray([Float(0),1,2,3]).reshaped(1,1,4,1)
        let call = try cache.append(keys:first,values:first,indexer:nil)
        XCTAssertEqual(call.0.dim(2),4)
        XCTAssertEqual(cache.state[0].asArray(Float.self),[2,3])
        XCTAssertEqual(cache.offset,4); XCTAssertEqual(cache.keyOffset,2)
        let next = MLXArray([Float(4)]).reshaped(1,1,1,1)
        let second = try cache.append(keys:next,values:next,indexer:nil)
        XCTAssertEqual(second.0.asArray(Float.self),[2,3,4])
        XCTAssertFalse(cache.isTrimmable); XCTAssertEqual(cache.trim(1),0)
        XCTAssertEqual(cache.offset,5)
        let truncated = cache.state.map { $0[0...,0...,..<1,0...] }
        XCTAssertFalse(cache.restoreDiskCacheState(truncated,metadata:cache.metaState,offset:5))
        XCTAssertEqual(cache.state.map { $0.dim(2) }, [2,2])
        XCTAssertEqual(cache.offset,5)
        let allowed = NaiveN05FlashMath.allowedMask(padding:MLXArray.ones([1,5],dtype:.bool),queryOffset:4,length:1,keyOffset:2,keyLength:3,window:3)
        XCTAssertEqual(allowed.asArray(Bool.self),[true,true,true])
            }
    }

    func testSparseRestoreIsAtomicAndTrimsCompanionTogether() throws {
        try MLXMetalTestLock.withLock {
        let cache = NaiveN05FlashCache(window:nil,requiresIndexer:true)
        let k=MLXArray.ones([1,1,3,4]),v=MLXArray.ones([1,1,3,2]),index=MLXArray.ones([1,1,3,4])
        _ = try cache.append(keys:k,values:v,indexer:index)
        let snapshot = cache.copy() as! NaiveN05FlashCache
        XCTAssertFalse(cache.restoreDiskCacheState([k,v],metadata:cache.metaState,offset:3))
        XCTAssertEqual(cache.offset,3);XCTAssertEqual(cache.state.count,3)
        XCTAssertThrowsError(try cache.append(keys:k,values:v,indexer:nil))
        XCTAssertEqual(cache.trim(2),2);XCTAssertEqual(cache.state.map{$0.dim(2)},[1,1,1])
        XCTAssertEqual(snapshot.offset,3);XCTAssertEqual(snapshot.state.map{$0.dim(2)},[3,3,3])
        let restored=NaiveN05FlashCache(window:nil,requiresIndexer:true)
        XCTAssertTrue(restored.restoreDiskCacheState(snapshot.state,metadata:snapshot.metaState,offset:3))
            }
    }

    func testRouterCorrectionSelectsButDoesNotChangeProbabilities() throws {
        try MLXMetalTestLock.withLock {
        let router = NaiveN05FlashRouter(try tiny())
        let weight = MLXArray([Float(0),0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,
            2,0,0,0,0,0,0,0, 3,0,0,0,0,0,0,0]).reshaped(4,8)
        try router.update(parameters: ModuleParameters.unflattened([
            "weight": weight, "e_score_correction_bias": MLXArray([Float(5),0,0,0]),
        ]), verify: [.all])
        let input = MLXArray([Float(1),0,0,0,0,0,0,0]).reshaped(1,1,8).asType(.bfloat16)
        let (indices, weights) = router(input)
        XCTAssertEqual(weights.dtype, .float32)
        XCTAssertEqual(Set(indices.asArray(Int32.self)), Set([Int32(0),3]))
        let probabilities = sigmoid(MLXArray([Float(0),1,2,3]))
        let expected = take(probabilities, indices, axis:0)
        close(weights, expected / expected.sum(axis:-1,keepDims:true))
            }
    }

    func testCustomFactoryConstructsNoAffineExpertBanks() throws {
        try MLXMetalTestLock.withLock {
        final class ConstructionProbe: Module, WeightedRoutedExpertLayer {
            func callRouted(_ input: MLXArray, indices: MLXArray, scores: MLXArray) -> MLXArray {
                XCTFail("Construction-only probe must not execute")
                return input
            }
        }
        var constructed = [Int]()
        let model = try NaiveN05FlashModel(tiny(), routedFactory: { layer, _ in
            constructed.append(layer)
            return ConstructionProbe()
        })
        XCTAssertEqual(constructed, [1])
        let names = model.parameters().flattened().map { $0.0 }
        XCTAssertFalse(names.contains { $0.contains("mlp.experts.") })
        XCTAssertTrue(names.contains("model.layers.1.mlp.gate.weight"))
            }
    }

    func testTinyRealModelFullChunkAndDecodeParityWithLeftPadding() throws {
        try MLXMetalTestLock.withLock {
        MLXRandom.seed(20260928)
        let model=try NaiveN05FlashModel(tiny())
        let tokens=MLXArray([0,0,3,4,5,6,7,8]).reshaped(1,8)
        let padding=MLXArray([false,false,true,true,true,true,true,true]).reshaped(1,8)
        let reference=try model(tokens,padding:padding)
        let cache=model.newCache()
        var chunks=[MLXArray]()
        for range in [0..<3,3..<5,5..<6,6..<8] {
            chunks.append(try model(tokens[0...,range],padding:padding[0...,..<range.upperBound],cache:cache))
        }
        close(concatenated(chunks,axis:1),reference,tolerance:3e-4)
        XCTAssertEqual(cache.map(\.offset),[8,8])
        XCTAssertEqual(cache[0].state.map{$0.dim(2)},[8,8,8])
        XCTAssertEqual(cache[1].state.map{$0.dim(2)},[2,2])
            }
    }
}
