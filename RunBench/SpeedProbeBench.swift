// BENCH_SPEED=1 — Swift twin of the Python campaign probe `mtp_anatomy_probe.py` (vMLX Python, 2026-10-04..06),
// so Swift and Python speeds on the same machine and bundle compare like for like.
//
// Same prompts (prose x3, code x3, easy_code x2, easy_prose x2), greedy, thinking off, max 300 tokens,
// one 16-token warm-up discarded. Decode tok/s uses engine generation tokens / generateTime;
// chunk timestamps are diagnostic only because streaming chunks may be coalesced.
//
// Production path: BatchEngine.generate (the host chat window's route -> solo fast path for MTP/DFlash2).
// Arms (BENCH_SPEED_ARM):
//   default   bundle-aware production selection (the default when no arm is supplied)
//   ar        explicit plain decode
//   adaptive  .nativeMTP(depth: BENCH_SPEED_MTP_DEPTH=3) + .adaptive(maximumDepth: BENCH_SPEED_MTP_MAX=5),
//             the Osaurus app's Adaptive mapping (MLXBatchAdapter.nativeMTPDepthPolicy)
//   dflash2   .dflash2(drafterPath: <bundle>/dflash2, blockSize: BENCH_SPEED_DFLASH2_BLOCK or nil)
// Output: one JSON line per prompt plus a summary (medians) to stdout and BENCH_SPEED_OUT.

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import VMLXTokenizers

private let speedPrompts: [(String, [String])] = [
    (
        "prose",
        [
            "Write a detailed essay of about 400 words on the history of the printing press and its effect on literacy in Europe.",
            "Write a detailed essay of about 400 words on how coral reefs form, why they bleach, and what restoration projects have learned.",
            "Write a detailed essay of about 400 words on the economics of medieval Venetian trade with the eastern Mediterranean.",
        ]
    ),
    (
        "code",
        [
            "Write a Python class implementing an LRU cache with get(key) and put(key, value) in O(1) using an OrderedDict, then a small unittest covering eviction order. Code only, no prose.",
            "Write a Rust function that parses an ISO-8601 date string into (year, month, day) with error handling, plus three unit tests. Code only, no prose.",
            "Write a TypeScript debounce(fn, ms) utility with leading/trailing options and cancel(), plus Jest tests for both modes. Code only, no prose.",
        ]
    ),
    (
        "easy_code",
        [
            "Write Python code that prints the numbers 1 to 100, one print statement per line, no loops. Code only.",
            "Write a JavaScript array literal containing the integers 1 through 150 in order. Code only.",
        ]
    ),
    (
        "easy_prose",
        [
            "Write the sentence: The quick brown fox jumps over the lazy dog. Write it 25 times, one per line.",
            "List the days of the week in order, Monday to Sunday, ten times, separated by commas.",
        ]
    ),
]

func runSpeedProbe(modelPath: String) async throws {
    let env = ProcessInfo.processInfo.environment
    let modelDir = URL(fileURLWithPath: modelPath)
    let arm = env["BENCH_SPEED_ARM"] ?? "default"
    let configData = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
    let jang = try? JangLoader.loadConfig(at: modelDir)
    let status = try? MTPBundleInspector.inspect(modelDirectory: modelDir, jangConfig: jang)
    let settings = VMLXServerRuntimeSettings()
    let defaultStrategy = settings.resolvedMTPDraftStrategy(
        configData: configData, jangConfig: jang, status: status, bundleDirectory: modelDir)
    let defaultLoad = settings.resolvedLoadConfiguration(
        configData: configData, jangConfig: jang, status: status)
    let maxTokens = Int(env["BENCH_SPEED_MAXTOK"] ?? "300") ?? 300
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context: ModelContext
    if arm == "default" {
        context = try await MLXLMCommon.loadModel(
            from: modelDir, using: #huggingFaceTokenizerLoader(),
            loadConfiguration: defaultLoad
        ).0
    } else if arm == "adaptive" {
        context = try await MLXLMCommon.loadModel(
            from: modelDir, using: #huggingFaceTokenizerLoader(),
            loadConfiguration: LoadConfiguration(nativeMTP: true)
        ).0
    } else {
        context = try await MLXLMCommon.loadModel(
            from: modelDir, using: #huggingFaceTokenizerLoader())
    }
    print(
        "[BENCH_SPEED_RESOLUTION] requested=\(arm) bundleDefault=\(String(describing: defaultStrategy))"
    )
    print(
        String(
            format: "[BENCH_SPEED] arm=%@ model=%@ load=%.1fs type=%@", arm,
            modelDir.lastPathComponent,
            CFAbsoluteTimeGetCurrent() - loadStart, String(describing: type(of: context.model))))
    nonisolated(unsafe) let ctx = context
    let engine = BatchEngine(context: ctx, maxBatchSize: 1, cacheCoordinator: nil)

    func params(_ budget: Int) -> GenerateParameters {
        var p = GenerateParameters(
            maxTokens: budget, temperature: 0, topP: 1, topK: 0, minP: 0,
            repetitionPenalty: nil)
        switch arm {
        case "default":
            p.draftStrategy = defaultStrategy
            if defaultStrategy?.nativeMTPDepth != nil {
                p.nativeMTPDepthPolicy = .adaptive(maximumDepth: 5)
            }
        case "adaptive":
            let depth = Int(env["BENCH_SPEED_MTP_DEPTH"] ?? "3") ?? 3
            let maxDepth = Int(env["BENCH_SPEED_MTP_MAX"] ?? "5") ?? 5
            p.draftStrategy = .nativeMTP(depth: depth)
            p.nativeMTPDepthPolicy = .adaptive(maximumDepth: maxDepth)
        case "dflash2":
            let block = env["BENCH_SPEED_DFLASH2_BLOCK"].flatMap(Int.init)
            p.draftStrategy = .dflash2(
                drafterPath: modelDir.appendingPathComponent("dflash2"), blockSize: block)
        default:
            break
        }
        return p
    }

    struct Row {
        var cat: String
        var tokens: Int
        var decode: Double
        var ttft: Double
        var text: String
        var chunkRate: Double
    }
    func run(_ prompt: String, _ budget: Int, cat: String) async throws -> Row {
        var input = UserInput(prompt: prompt)
        input.additionalContext = ["enable_thinking": false]
        let t0 = CFAbsoluteTimeGetCurrent()
        let prepared = try await ctx.processor.prepare(input: input)
        nonisolated(unsafe) let send = prepared
        let stream = await engine.generate(input: send, parameters: params(budget))
        var first: Double?
        var last = 0.0
        var tokens = 0
        var genTime = 0.0
        var mtp = ""
        var text = ""
        for await ev in stream {
            switch ev {
            case .chunk(let c):
                let now = CFAbsoluteTimeGetCurrent()
                if first == nil { first = now }
                last = now
                text += c
            case .reasoning(let r):
                let now = CFAbsoluteTimeGetCurrent()
                if first == nil { first = now }
                last = now
                text += r
            case .info(let info):
                tokens = info.generationTokenCount
                genTime = info.generateTime
                if let m = info.nativeMTPStats {
                    mtp = String(
                        format:
                            "verify=%d commit/verify=%.2f arFallback=%d accByDepth=%@ depth=%d->%d downshifts=%d arTrips=%d/%d mode=%@",
                        m.verifyCalls, m.avgCommittedPerVerify, m.arFallbackTokens,
                        m.acceptedByDepth.description, m.depth, m.activeDepth, m.adaptiveDownshifts,
                        m.arSafetyTrips, m.arSafetyResumes, m.verifierMode)
                }
            default:
                break
            }
        }
        // Primary: the engine's own generation clock (tokens / generateTime, prompt excluded) — chunk
        // stamps are unusable here because the host stream coalesces many tokens into one chunk.
        let decode = genTime > 0 ? Double(tokens) / genTime : 0
        let span = (first.map { last - $0 }) ?? 0
        let chunkRate = span > 0.2 && tokens > 1 ? Double(tokens - 1) / span : 0
        if !mtp.isEmpty { print("[BENCH_SPEED_MTP] \(cat) \(mtp)") }
        return Row(
            cat: cat, tokens: tokens, decode: decode, ttft: (first ?? t0) - t0, text: text,
            chunkRate: chunkRate)
    }

    _ = try await run("Say hi.", 16, cat: "warmup")
    var summary: [String: [Double]] = [:]
    var lines: [String] = []
    var texts: [String] = []
    // BENCH_SPEED_CATS=prose,code — run a subset (short runs from a cool start; see the handoff's clock notes).
    let selected = env["BENCH_SPEED_CATS"].map { Set($0.split(separator: ",").map(String.init)) }
    for (cat, prompts) in speedPrompts where selected?.contains(cat) ?? true {
        for prompt in prompts {
            let r = try await run(prompt, maxTokens, cat: cat)
            summary[cat, default: []].append(r.decode)
            let preview = String(r.text.prefix(80)).replacingOccurrences(of: "\n", with: "\\n")
            let line = String(
                format:
                    "{\"cat\":\"%@\",\"tokens\":%d,\"decode_tok_s\":%.2f,\"chunk_tok_s\":%.2f,\"ttft\":%.3f,\"preview\":%@}",
                cat, r.tokens, r.decode, r.chunkRate, r.ttft, String(reflecting: preview))
            print("[BENCH_SPEED_ROW] " + line)
            lines.append(line)
            texts.append(r.text)
        }
    }
    func median(_ v: [Double]) -> Double {
        let s = v.sorted()
        return s.isEmpty
            ? 0 : (s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2)
    }
    let med = speedPrompts.filter { selected?.contains($0.0) ?? true }.map { cat, _ in
        String(format: "\"%@\": %.2f", cat, median(summary[cat] ?? []))
    }
    .joined(separator: ", ")
    print("[BENCH_SPEED_SUMMARY] arm=\(arm) {\(med)}")
    if let textOut = env["BENCH_SPEED_TEXT_OUT"] {
        // Full generated texts in prompt order (greedy identity check across arms).
        let data = try JSONSerialization.data(withJSONObject: texts, options: [.prettyPrinted])
        try data.write(to: URL(fileURLWithPath: textOut))
    }
    if let out = env["BENCH_SPEED_OUT"] {
        try (lines + ["{\"summary\": {\(med)}}"]).joined(separator: "\n").write(
            toFile: out, atomically: true, encoding: .utf8)
    }
}
