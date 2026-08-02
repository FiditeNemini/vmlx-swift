// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// `drop_thinking` vs prefix caching.
//
// Prefix reuse requires that turn N's rendered prompt stay a byte prefix of
// turn N+1's. `dropThinkingMessages` strips `reasoning_content` from every
// assistant message BEFORE the last user message (mirroring
// `_drop_thinking_messages` in encoding_dsv4.py), so each new user turn pushes
// the previous assistant turn across that line and REMOVES its reasoning from
// the history that was already cached. The bytes change retroactively and
// every boundary stored before that turn stops matching.
//
// Live DSV4 evidence (isolated Release root, VMLX_CACHE_FETCH_TRACE=1): only
// the immediately-preceding store ever hit —
//   fetch tokens=2178 -> HIT boundary=2084   (stored by the previous turn)
//   fetch tokens=2086 -> MISS, although 1829 and 207 were both stored
// so every turn paid a near-full cold prefill and TTFT climbed 17.32s ->
// 28.86s across consecutive turns.

import Foundation
import MLXLMCommon
import Testing

@Suite("DSV4 drop_thinking vs prefix reuse")
struct DeepseekV4DropThinkingCacheTests {

    private func tools() -> [[String: any Sendable]] {
        [
            [
                "type": "function",
                "function": [
                    "name": "file_write",
                    "parameters": [
                        "type": "object",
                        "properties": ["path": ["type": "string"] as [String: any Sendable]],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable]
        ]
    }

    /// Two consecutive renders of the same growing conversation.
    private func renders(
        withTools: Bool
    ) throws -> (turnN: String, turnNPlus1: String) {
        var base: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is a KV cache?"],
            [
                "role": "assistant", "content": "A KV cache stores attention tensors.",
                "reasoning_content": "The user wants a short definition. Keep it to two sentences.",
            ],
        ]
        let turnN = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: base,
            tools: withTools ? tools() : nil,
            additionalContext: ["enable_thinking": true],
            addGenerationPrompt: true)

        // The next user turn arrives. Nothing earlier was edited by the caller.
        base.append(["role": "user", "content": "And what is a Bloom filter?"])
        let turnNPlus1 = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: base,
            tools: withTools ? tools() : nil,
            additionalContext: ["enable_thinking": true],
            addGenerationPrompt: true)
        return (turnN, turnNPlus1)
    }

    @Test("without tools, a new user turn retroactively strips earlier reasoning")
    func withoutToolsReasoningIsStrippedRetroactively() throws {
        let (turnN, turnNPlus1) = try renders(withTools: false)

        // Turn N rendered the reasoning; turn N+1 dropped it, so the earlier
        // prompt is NOT a prefix of the later one and every cache entry stored
        // for turn N is unusable.
        #expect(turnN.contains("The user wants a short definition"))
        #expect(!turnNPlus1.contains("The user wants a short definition"))

        let sharedPrefix = zip(turnN, turnNPlus1).prefix { $0 == $1 }.count
        #expect(
            sharedPrefix < turnN.count,
            "expected the histories to diverge once reasoning is dropped")
    }

    @Test("with tools, reasoning is retained so the history stays append-only")
    func withToolsHistoryStaysAppendOnly() throws {
        let (turnN, turnNPlus1) = try renders(withTools: true)

        // The official contract disables drop_thinking when tools are present,
        // which is also what makes prefix reuse possible across an agent loop.
        #expect(turnN.contains("The user wants a short definition"))
        #expect(turnNPlus1.contains("The user wants a short definition"))

        // Everything up to turn N's trailing generation rail must survive
        // verbatim, otherwise the boundary stored for turn N cannot be reused.
        let railless = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: [
                ["role": "system", "content": "You are a helpful assistant."],
                ["role": "user", "content": "What is a KV cache?"],
                [
                    "role": "assistant", "content": "A KV cache stores attention tensors.",
                    "reasoning_content":
                        "The user wants a short definition. Keep it to two sentences.",
                ],
            ],
            tools: tools(),
            additionalContext: ["enable_thinking": true],
            addGenerationPrompt: false)
        #expect(
            turnNPlus1.hasPrefix(railless),
            "turn N's history must remain a byte prefix of turn N+1")
    }
}
