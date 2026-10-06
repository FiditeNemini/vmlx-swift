// BENCH_ROWCOST=1 — target forward cost by row count (Swift twin of vMLX Python `step_ms_rows.py`).
// Loads with native MTP, prefills a ~500-token prompt, then for each row count R times one forward of R new
// tokens from a COPY of the prefilled cache (so every repetition starts from the same state), median of
// BENCH_ROWCOST_REPS (default 9) after 3 warm-ups.
//   exact   : R=1 nativeBackboneForward, R>1 nativeBackboneMTPVerifyForward (the iterator's verify entry,
//             FlashVerificationScope row-exact routes active)
//   plain   : nativeBackboneForward for every R (ordinary multi-row forward, no verification scope)
// Prints one line per R: "[ROWCOST] mode=exact rows=R ms=…".

import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import VMLXTokenizers

func runRowCostBench(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let modelDir = URL(fileURLWithPath: modelPath)
    let reps = Int(env["BENCH_ROWCOST_REPS"] ?? "9") ?? 9
    let rowsList = (env["BENCH_ROWCOST_ROWS"] ?? "1,2,3,4,5,6,8").split(separator: ",").compactMap { Int($0) }
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader(),
        loadConfiguration: LoadConfiguration(nativeMTP: true)).0
    guard let model = context.model as? any NativeMTPModel else {
        print("[ROWCOST] model is not a NativeMTPModel"); return
    }
    let text = String(repeating: "The history of printing presses in Europe changed literacy and trade. ", count: 40)
    var input = UserInput(prompt: text)
    input.additionalContext = ["enable_thinking": false]
    let prepared = try await context.processor.prepare(input: input)
    let prompt = prepared.text.tokens.reshaped(1, -1)
    let base = model.newCache(parameters: GenerateParameters())
    let pre = model.nativeBackboneForward(prompt, cache: base)
    MLX.eval(pre.logits, pre.hiddenStates)
    MLX.eval(base)
    print("[ROWCOST] model=\(modelDir.lastPathComponent) prompt=\(prompt.dim(1)) tokens")

    for mode in (env["BENCH_ROWCOST_MODES"] ?? "exact,plain").split(separator: ",").map(String.init) {
        for rows in rowsList {
            var samples: [Double] = []
            var builds: [Double] = []
            for rep in 0..<(reps + 3) {
                let cache = base.map { $0.copy() }
                MLX.eval(cache)
                let ids = MLXArray((0..<rows).map { Int32(1000 + (($0 * 7919 + rep * 31) % 20000)) })
                    .reshaped(1, rows)
                let t0 = CFAbsoluteTimeGetCurrent()
                let out: NativeMTPForwardResult
                if mode == "exact" && rows > 1 {
                    // The iterator's actual verify mode (NativeMTPTokenIterator.verifyCycle, staged path).
                    out = NativeMTPVerifierStatePolicy.withVerifierMode("input_capture_staged") {
                        model.nativeBackboneMTPVerifyForward(ids, cache: cache)
                    }
                } else {
                    out = model.nativeBackboneForward(ids, cache: cache)
                }
                let built = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                MLX.eval(out.logits, out.hiddenStates)
                let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                if rep >= 3 { samples.append(ms); builds.append(built) }
            }
            samples.sort(); builds.sort()
            print(String(format: "[ROWCOST] mode=%@ rows=%d ms=%.2f min=%.2f build=%.2f", mode, rows,
                         samples[samples.count / 2], samples[0], builds[builds.count / 2]))
        }
    }
}
