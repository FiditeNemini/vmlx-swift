// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@Suite("DSV4 agentic tool source contracts")
struct DSV4AgenticToolSourceTests {
    @Test("agentic row enforces DSML, exact args, native interleaving, defaults, and disk-only L2")
    func agenticRowEnforcesDSMLThinkParserExactArgsDefaultsAndDiskOnlyL2() throws {
        let bench = try String(contentsOfFile: "RunBench/Bench.swift", encoding: .utf8)

        #expect(bench.contains("BENCH_DSV4_AGENTIC_TOOL"))
        #expect(bench.contains("func runDSV4AgenticToolCheck("))
        #expect(bench.contains("context.configuration.toolCallFormat == .dsml"))
        #expect(bench.contains("ReasoningParser.fromCapabilityName("))
        #expect(bench.contains("reasoningParser?.startTag == \"<think>\""))
        #expect(bench.contains("reasoningParser?.endTag == \"</think>\""))
        #expect(bench.contains("BENCH_DSV4_AGENTIC_CACHE_PAGED\"] == \"1\""))
        #expect(bench.contains("toolCall.function.arguments[\"launch_id\"] == .string(launchID)"))
        #expect(bench.contains("toolTurn.toolProgressDeltas > 0"))
        #expect(bench.contains("generationConfig: ctx.configuration.generationDefaults"))
        #expect(bench.contains("BENCH_DSV4_AGENTIC_INTERLEAVED"))
        #expect(bench.contains("reasoningContent: toolTurn.reasoning.isEmpty ? nil : toolTurn.reasoning"))
        #expect(bench.contains("chat.append(.tool(statusResultJSON, toolCallId: toolCallId))"))
        #expect(bench.contains("reasoning->tool1->result1->reasoning->tool2->result2"))
        #expect(bench.contains("->reasoning->final->thinkingOffFollowup"))
        #expect(!bench.contains("Now use \\(windowToolName)"))
        #expect(bench.contains("markerLeaks(in: result.text)"))
        #expect(bench.contains("snapshot.isPagedIncompatible"))
        #expect(bench.contains("diskStats.stores > 0"))
        #expect(bench.contains("diskStats.hits > 0"))
        #expect(!bench.contains("BENCH_DSV4_AGENTIC_TEMP\"] ??"))
        #expect(!bench.contains("BENCH_DSV4_AGENTIC_REPETITION_PENALTY\"] ??"))
    }
}
