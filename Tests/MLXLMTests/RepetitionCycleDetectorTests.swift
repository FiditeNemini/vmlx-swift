import Foundation
import Testing

@testable import MLXLMCommon

/// The guard has to fire on the observed collapse and stay silent on every
/// ordinary shape that happens to repeat. False positives here truncate real
/// answers, so the negative cases matter more than the positive one.
@Suite("Degenerate repetition detector")
struct RepetitionCycleDetectorTests {

    /// Verbatim from the live Raptor turn that spent its whole token budget.
    static let observedUnit =
        "The answer is AppleScript; it begins with `use AppleScript version`. "
        + "I do not generate or repeat the request. "

    @Test("fires on the observed collapse")
    func detectsObservedLoop() throws {
        let text = String(repeating: Self.observedUnit, count: 6)
        let cycle = try #require(RepetitionCycleDetector.cycle(in: text))
        #expect(cycle.unit == Self.observedUnit)
        #expect(cycle.repeats >= RepetitionCycleDetector.minimumRepeats)
    }

    @Test("reports the shortest primitive unit, not a multiple of it")
    func reportsShortestUnit() throws {
        // Primitive: no shorter string repeats to build it.
        let unit = "the same sentence again and again, ok? "  // 39 chars
        let cycle = try #require(RepetitionCycleDetector.cycle(in: String(repeating: unit, count: 8)))
        #expect(cycle.unit == unit, "reported \(cycle.unit.count)-char unit: \(cycle.unit.debugDescription)")
    }

    /// A unit that is itself built from a shorter period is filler, however
    /// long it looks — `"abcdefgh" x 4` is a 32-character unit with period 8.
    @Test("a non-primitive unit is treated as filler")
    func nonPrimitiveUnitIsFiller() {
        let unit = String(repeating: "abcdefgh", count: 4)
        #expect(RepetitionCycleDetector.cycle(in: String(repeating: unit, count: 8)) == nil)
        #expect(RepetitionCycleDetector.minimalPeriod(of: Array(unit)) == 8)
    }

    @Test("streams to the same verdict as the whole-string form")
    func streamingMatchesWholeString() throws {
        var detector = RepetitionCycleDetector()
        var fired: RepetitionCycleDetector.Cycle?
        for _ in 0 ..< 8 {
            // Arrive in small pieces, the way detokenized chunks do.
            for piece in Self.observedUnit.chunked(into: 7) where fired == nil {
                fired = detector.feed(piece)
            }
        }
        let cycle = try #require(fired)
        #expect(cycle.unit == Self.observedUnit)
    }

    // MARK: - Must NOT fire

    @Test("three repeats are not enough")
    func threeRepeatsAreBelowThreshold() {
        #expect(RepetitionCycleDetector.cycle(in: String(repeating: Self.observedUnit, count: 3)) == nil)
    }

    @Test(
        "short repeating filler never qualifies",
        arguments: [
            String(repeating: "-", count: 400),
            String(repeating: "= ", count: 200),
            String(repeating: "...", count: 140),
            String(repeating: "| | |\n", count: 80),
            String(repeating: "\n", count: 500),
        ])
    func shortFillerIsIgnored(_ text: String) {
        #expect(
            RepetitionCycleDetector.cycle(in: text) == nil,
            "fired on filler: \(text.prefix(12).debugDescription)")
    }

    @Test("ordinary prose does not fire")
    func proseIsIgnored() {
        let prose = """
            A Merkle proof lets a verifier confirm that one leaf belongs to a tree \
            without downloading the whole tree. The prover supplies the audit path: \
            the sibling hash at each level between the leaf and the root. The verifier \
            hashes upward and compares the result against the known root. If they \
            match, the leaf is in the tree; if not, it is not. The cost is logarithmic \
            in the number of leaves rather than linear, which is what makes the scheme \
            practical for large datasets and for clients that cannot store them.
            """
        #expect(RepetitionCycleDetector.cycle(in: prose) == nil)
    }

    /// Repeated *structure* with differing content is the shape most at risk of
    /// a false positive, and it must survive.
    @Test("a table with distinct rows does not fire")
    func tableWithDistinctRowsIsIgnored() {
        var table = "| name | size | kind |\n| --- | --- | --- |\n"
        for i in 0 ..< 40 {
            table += "| entry_\(i) | \(i * 137) bytes | file |\n"
        }
        #expect(RepetitionCycleDetector.cycle(in: table) == nil)
    }

    @Test("repeated code blocks with differing bodies do not fire")
    func codeWithDifferingBodiesIsIgnored() {
        var code = ""
        for i in 0 ..< 30 {
            code += "func step\(i)() -> Int {\n    return \(i) * 3 + 1\n}\n\n"
        }
        #expect(RepetitionCycleDetector.cycle(in: code) == nil)
    }

    @Test("short output is never examined")
    func shortOutputIsIgnored() {
        var detector = RepetitionCycleDetector()
        #expect(detector.feed(String(repeating: "xy", count: 20)) == nil)
    }

    @Test("disabled detector never fires")
    func disabledNeverFires() {
        var detector = RepetitionCycleDetector(isEnabled: false)
        #expect(detector.feed(String(repeating: Self.observedUnit, count: 10)) == nil)
    }

    @Test("VMLX_REPETITION_STOP=0 disables it")
    func environmentOptOut() {
        #expect(!RepetitionCycleDetector.fromEnvironment(["VMLX_REPETITION_STOP": "0"]).isEnabled)
        #expect(RepetitionCycleDetector.fromEnvironment([:]).isEnabled)
    }
}

extension String {
    fileprivate func chunked(into size: Int) -> [String] {
        var out: [String] = []
        var index = startIndex
        while index < endIndex {
            let next = self.index(index, offsetBy: size, limitedBy: endIndex) ?? endIndex
            out.append(String(self[index ..< next]))
            index = next
        }
        return out
    }

    // MARK: - Line cycles (units longer than the character scan's 512 bound)

    /// The gap the line scan exists for: a repeating unit LONGER than the
    /// character scan's `maximumUnitLength` (512).
    ///
    /// A single repeated line under 512 characters is ALREADY caught by the
    /// character scan at 4 repeats — verified by `firesOnObservedCollapse`.
    /// The escape hatch is a longer block, where `cycle(in:)` deliberately
    /// gives up ("cheaper to let the token cap handle"). That fallback costs
    /// ~2 minutes on a fast Mac and ~34 minutes at 8 tok/s on a base laptop,
    /// which is the live report this closes.
    @Test("a repeating block LONGER than the character scan's bound is caught")
    func blockLongerThanCharacterBoundIsCaught() throws {
        // Six substantial lines: comfortably past 512 characters as a unit,
        // so the character scan cannot see the period.
        let block = [
            "First, open the depot manifest and confirm every column header matches the schema.",
            "Second, normalise each region code against the canonical lookup table in memory.",
            "Third, recompute the seal codes for any row whose port number changed since import.",
            "Fourth, write the corrected rows back out in the original order, preserving gaps.",
            "Fifth, verify the checksum of the rewritten file against the manifest total.",
            "Sixth, report how many rows changed and list the identifiers that were touched.",
            "Seventh, archive the previous revision alongside a timestamped audit record.",
            "Eighth, release the advisory lock and emit the completion summary to the log.",
        ]
        let unitLength = block.joined(separator: "\n").count
        #expect(
            unitLength > RepetitionCycleDetector.maximumUnitLength,
            "the unit must exceed the character scan's bound or this tests nothing")

        var detector = RepetitionCycleDetector()
        var fired: RepetitionCycleDetector.Cycle?
        outer: for _ in 0 ..< 12 {
            for line in block {
                if let cycle = detector.feed(line + "\n") { fired = cycle; break outer }
            }
        }
        let cycle = try #require(fired, "a >512-char block repeated 12 times must be caught")
        #expect(cycle.repeats >= RepetitionCycleDetector.minimumLineRepeats)
    }

    /// A short repeating block (under 512 characters) is caught too — by the
    /// CHARACTER scan, not the line scan. Kept to pin that the short case did
    /// not regress when the line path was added.
    @Test("a short repeating multi-line block is still caught")
    func repeatingBlockIsCaught() throws {
        let block = [
            "Step one: read the manifest and validate every column header.",
            "Step two: normalise the region codes against the lookup table.",
            "Step three: write the corrected rows back to the output file.",
        ]
        var detector = RepetitionCycleDetector()
        var fired: RepetitionCycleDetector.Cycle?
        for _ in 0 ..< 10 {
            for line in block {
                if let cycle = detector.feed(line + "\n") { fired = cycle; break }
            }
            if fired != nil { break }
        }
        #expect(fired != nil, "a 3-line block repeated 10 times must be caught")
    }

    /// FALSE POSITIVES ARE WORSE THAN THE BUG. Formatting that legitimately
    /// repeats must never halt a generation.
    @Test("ordinary repeated formatting never fires")
    func formattingDoesNotFire() {
        let benign: [[String]] = [
            Array(repeating: "|---|---|---|", count: 40),          // table rule
            Array(repeating: "", count: 40),                        // blank lines
            Array(repeating: "    }", count: 40),                   // closers
            Array(repeating: "- [ ] todo", count: 40),              // short list items
            Array(repeating: "====================", count: 40),    // separators
        ]
        for lines in benign {
            var detector = RepetitionCycleDetector()
            var fired: RepetitionCycleDetector.Cycle?
            for line in lines where fired == nil {
                fired = detector.feed(line + "\n")
            }
            #expect(fired == nil, "benign formatting fired: \(lines[0])")
        }
    }

    /// Near-repetition is NOT degeneration: a real script whose lines differ
    /// by an index is doing useful work and must run to completion.
    @Test("lines that vary are never treated as a cycle")
    func varyingLinesDoNotFire() {
        var detector = RepetitionCycleDetector()
        var fired: RepetitionCycleDetector.Cycle?
        for i in 0 ..< 60 where fired == nil {
            fired = detector.feed("        data[\(i)] = transform(row_\(i), mode=\"strict\")\n")
        }
        #expect(fired == nil, "varying lines must not be reported as degenerate")
    }

    /// A unit that is itself a repetition has the shorter block as its real
    /// period — `A A` must never be reported as a period-2 cycle.
    @Test("a doubled unit reports its primitive period")
    func doubledUnitIsNotPeriodTwo() throws {
        let line = "The answer is AppleScript; it begins with `use AppleScript version`."
        let lines = Array(repeating: line, count: 20)
        let cycle = try #require(RepetitionCycleDetector.lineCycle(in: lines))
        #expect(cycle.unit == line, "expected the primitive single line, got a multiple")
    }

    /// Disabling must restore exactly the old behaviour on the line path too.
    @Test("VMLX_REPETITION_STOP=0 disables the line scan as well")
    func disabledSkipsLineScan() {
        var detector = RepetitionCycleDetector.fromEnvironment(["VMLX_REPETITION_STOP": "0"])
        var fired: RepetitionCycleDetector.Cycle?
        for _ in 0 ..< 30 where fired == nil {
            fired = detector.feed("        data[index] = transform(row, mode=strict, retries=3)\n")
        }
        #expect(fired == nil)
    }
}
