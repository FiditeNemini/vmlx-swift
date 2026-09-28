# Preserve activation precision during optional tied-head quantization

Optional load-time tied-head quantization converted an FP16 embedding to a quantized embedding, then the preserved-affine loader policy forced its output to BF16. In Gemma 4 QAT, FP16 router and layer scales mixed with these activations and promoted arithmetic to FP32. Quantizing storage therefore changed the model activation precision and reduced throughput.

The conversion now records the original floating embedding dtype. The later inferred alignment policy respects an explicit dtype while retaining the existing BF16 alignment for checkpoint-quantized embeddings without an explicit contract. Sampler, template, cache policy, and compilation settings are unchanged.

## Validation

- Release production and test targets built successfully.
- Two Metal regression tests passed: source FP16/BF16/FP32 lookup, layer-scale multiplication and tied projection retain their activation dtype with bounded embedding quantization error; checkpoint-quantized FP16 metadata still receives the existing BF16 alignment.
- On M5 Max, Gemma 4 26B-A4B QAT JANG_4M with a Q6 tied head, a 2,588-token prompt and bundle-default sampling measured 81.4,81.4,81.2 tok/s, versus 33.9,33.9,33.9 before the fix. One warmup preceded each three-run measurement. All measured answers were identical between the Q6 arms and stopped naturally. Reported physical footprint fell from 13,821 to 11,764 MiB. This is an engine-harness result, not a universal chip or application claim.
- Two concurrent requests returned the independent expected answers and stopped naturally with both slots overlapping. These short rows are correctness checks, not steady-state throughput measurements.
- Real image replay restored 297/298 prefix tokens; the actual-history follow-up restored 298/351. Changed media missed the cache and produced the changed image description. All four responses stopped naturally at approximately94–96 tok/s with coherent visible output.

## Outstanding app gate

Repin and build the consuming app, then prove Q6 with normal prefix/disk caching, visible multi-turn output, concurrent children and physical footprint through its actual GUI. The app gate remains open; no merge readiness or release claim.
