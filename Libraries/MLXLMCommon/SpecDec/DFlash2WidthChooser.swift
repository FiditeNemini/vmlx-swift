// Port of vMLX Python `dflash2_runtime._BlockChooser` / `_dflash2_block_plan` (commits 39166fc02, b299ab6dc;
// audit R2-20 §3 and R2-21 §5c). Chooses the DFlash2 verify width per cycle.
//
// WHY: a wider verify drafts more tokens but costs one wider target forward. With the lane matmul
// (LaneQMM.swift) 8 and 16 verify rows cost about the same, so width 16 pays on predictable text
// (+51-60 % easy text in Python) while prose, where deep draft positions rarely land, wants 8.
// Fixed widths lose somewhere; the rate-EMA and acceptance-only rules were measured and rejected in Python.
//
// MODEL (expected tokens per measured second, TensorFold's depth objective):
//   E[tokens | w] = 1 + sum_{j=1}^{w-1} prod_{k<=j} p_k, p_k = P(draft position k accepted | 1..k-1 were).
//   p_k is learned from EVERY cycle at ANY width (positions 1..accepted hit, accepted+1 missed if it was
//   verified), counts decay x0.85 per cycle; a position with little data leans on the previous estimate
//   (one pseudo-trial, prior 0.7), so width-8 cycles teach width 16's first 7 positions.
//   Cost per width = EMA (0.25) of measured cycle seconds, shared process-wide per target (a model+kernel
//   property, not a text property); the first cycle at each width in the process is discarded (kernel
//   compile / warm-up). The non-current width measured longest ago is re-timed every 32 cycles.
// Correctness: width only changes how many drafts are verified, never the acceptance rule.

import Foundation

final class DFlash2WidthCostTable: @unchecked Sendable {
    private let lock = NSLock()
    private var cost: [Int: Double] = [:]
    private var warm: Set<Int> = []

    func observe(width: Int, seconds: Double) {
        lock.lock()
        defer { lock.unlock() }
        if !warm.contains(width) {
            warm.insert(width)
            return
        }
        if let c = cost[width] {
            cost[width] = c + 0.25 * (seconds - c)
        } else {
            cost[width] = seconds
        }
    }

    func cost(of width: Int) -> Double? {
        lock.lock()
        defer { lock.unlock() }
        return cost[width]
    }

    nonisolated(unsafe) private static var tables: [ObjectIdentifier: DFlash2WidthCostTable] = [:]
    private static let tablesLock = NSLock()

    static func shared(for target: AnyObject, widths: [Int]) -> DFlash2WidthCostTable {
        tablesLock.lock()
        defer { tablesLock.unlock() }
        let key = ObjectIdentifier(target)
        if let t = tables[key] { return t }
        let t = DFlash2WidthCostTable()
        tables[key] = t
        return t
    }
}

struct DFlash2WidthChooser {
    static let decay = 0.85
    static let probeEvery = 32
    static let priorP = 0.7

    let widths: [Int]
    private(set) var width: Int
    private var trials: [Double]
    private var hits: [Double]
    private var cycles = 0
    private var lastTimed: [Int: Int] = [:]
    private let costs: DFlash2WidthCostTable

    /// Candidate widths: (trained, 2*trained) when the target's verify cost is lane-flat, else
    /// (min(5, trained), trained, 2*trained). `VMLX_DFLASH2_BLOCK=<n>` pins one width (A/B tool).
    static func plan(trained: Int, laneFlat: Bool) -> [Int] {
        let t = max(2, trained)
        if let raw = ProcessInfo.processInfo.environment["VMLX_DFLASH2_BLOCK"], let n = Int(raw) {
            return [max(2, min(n, 2 * t))]
        }
        // Width 1 = a plain AR step, measured like any other width: when speculation emits fewer
        // tokens per second than AR (27B JANGH2 sampled prose: 24 vs 29 tok/s), the chooser
        // picks AR; decayed acceptance drifting back to the prior plus the periodic re-time
        // re-probes speculation. `VMLX_DFLASH2_AR_OPTION=0` removes it.
        let ar = ProcessInfo.processInfo.environment["VMLX_DFLASH2_AR_OPTION"] == "0" ? [] : [1]
        if laneFlat { return ar + [t, 2 * t] }
        return ar + Array(Set([min(5, t), t, 2 * t])).sorted()
    }

    init(widths: [Int], costs: DFlash2WidthCostTable) {
        self.widths = widths.sorted()
        self.width = self.widths.first(where: { $0 > 1 }) ?? self.widths[0]
        let top = self.widths.last ?? 2
        self.trials = Array(repeating: 0, count: top)
        self.hits = Array(repeating: 0, count: top)
        self.costs = costs
    }

    func expectedTokens(_ w: Int) -> Double {
        var e = 1.0
        var run = 1.0
        var prev = Self.priorP
        if w > 1 {
            for k in 1 ..< w {
                prev = (hits[k] + prev) / (trials[k] + 1.0)
                run *= prev
                e += run
            }
        }
        return e
    }

    /// One finished cycle: `tokens` emitted (accepted + 1) in `seconds` wall time at `verifyWidth` rows.
    mutating func observe(verifyWidth: Int, tokens: Int, seconds: Double) {
        guard seconds > 0, widths.contains(verifyWidth) else { return }  // clipped final block
        let accepted = tokens - 1
        for k in 1 ..< trials.count {
            trials[k] *= Self.decay
            hits[k] *= Self.decay
        }
        let lastVerified = min(accepted + 1, verifyWidth - 1)
        if lastVerified >= 1 {
            for k in 1 ... lastVerified where k < trials.count {
                trials[k] += 1
                if k <= accepted { hits[k] += 1 }
            }
        }
        costs.observe(width: verifyWidth, seconds: seconds)
        cycles += 1
        lastTimed[verifyWidth] = cycles
        guard widths.count > 1 else { return }
        if let unseen = widths.first(where: { costs.cost(of: $0) == nil }) {
            width = unseen
        } else if cycles % Self.probeEvery == 0 {
            let others = widths.filter { $0 != verifyWidth }
            width = others.min { (lastTimed[$0] ?? -1) < (lastTimed[$1] ?? -1) } ?? verifyWidth
        } else {
            width =
                widths.max {
                    expectedTokens($0) / costs.cost(of: $0)! < expectedTokens($1) / costs.cost(
                        of: $1)!
                }
                ?? width
        }
    }
}
