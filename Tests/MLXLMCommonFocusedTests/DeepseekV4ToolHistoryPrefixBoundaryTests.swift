// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Live DSV4 agent rows lost prefix-cache reuse the moment a tool call entered
// the transcript. Captured from a real Osaurus run (minesweeper artifact):
//
//   prompt=1458 stable=[72, 1128] all=[72, 1128, 1456]   <- history boundary
//   prompt=7017 stable=[72, 1128] all=[72, 1128]         <- history boundary GONE
//   HIT disk boundary=1457 remaining=5560
//
// `canonicalChatCacheBoundaries` derives the history boundary by re-rendering
// the whole message list with `addGenerationPrompt: false` and requiring the
// result to be an exact TOKEN PREFIX of the real prompt. Once that re-render
// stops matching, every later agent turn cold-prefills the entire growing
// transcript, which is what made tool turns appear to hang forever.
//
// These tests pin that round-trip for tool histories.

import Foundation
import MLXLMCommon
import Testing

@Suite("DSV4 tool-history prefix boundary round-trip")
struct DeepseekV4ToolHistoryPrefixBoundaryTests {

    /// The exact rail contract the boundary derivation depends on: rendering
    /// without the generation prompt must be a strict textual prefix of
    /// rendering with it. Anything else costs the history boundary.
    private func expectPrefixRoundTrip(
        _ messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        label: String
    ) throws {
        let withRail = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: messages,
            tools: tools,
            additionalContext: nil,
            addGenerationPrompt: true)
        let withoutRail = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: messages,
            tools: tools,
            additionalContext: nil,
            addGenerationPrompt: false)

        guard withRail.hasPrefix(withoutRail) else {
            // Surface the exact divergence point — that byte is the bug.
            let common = zip(withRail, withoutRail).prefix { $0 == $1 }.count
            let railTail = String(withRail.dropFirst(common).prefix(120))
            let noRailTail = String(withoutRail.dropFirst(common).prefix(120))
            Issue.record(
                """
                \(label): no-generation-prompt render is NOT a prefix of the real prompt.
                diverges at char \(common)
                  withRail   : \(railTail.debugDescription)
                  withoutRail: \(noRailTail.debugDescription)
                """)
            return
        }
        #expect(withRail.count > withoutRail.count, "\(label): rail was not appended at all")
    }

    private func weatherTools() -> [[String: any Sendable]] {
        [
            [
                "type": "function",
                "function": [
                    "name": "file_write",
                    "description": "Write a file",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "path": ["type": "string"] as [String: any Sendable],
                            "content": ["type": "string"] as [String: any Sendable],
                        ] as [String: any Sendable],
                        "required": ["path", "content"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable]
        ]
    }

    @Test("plain chat history keeps the exact-prefix round trip")
    func plainChatHistoryRoundTrips() throws {
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a coding agent."],
            ["role": "user", "content": "Hi"],
            ["role": "assistant", "content": "Hello.", "reasoning_content": "greeting"],
            ["role": "user", "content": "Build me a minesweeper game"],
        ]
        try expectPrefixRoundTrip(messages, tools: weatherTools(), label: "plain chat")
    }

    @Test("a tool-call turn in history keeps the exact-prefix round trip")
    func toolCallHistoryRoundTrips() throws {
        // The shape that killed the boundary live: assistant emits a DSML tool
        // call carrying a large string payload, the result comes back, and the
        // next turn must still reuse everything before it.
        let payload = String(repeating: "<div class=\"cell\"></div>\n", count: 60)
        let arguments = String(
            data: try JSONSerialization.data(
                withJSONObject: ["path": "/tmp/minesweeper.html", "content": payload],
                options: [.sortedKeys]),
            encoding: .utf8)!

        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a coding agent."],
            ["role": "user", "content": "Build me a minesweeper game"],
            [
                "role": "assistant",
                "content": "",
                "reasoning_content": "I'll write the HTML file and share it.",
                "tool_calls": [
                    [
                        "id": "call_1",
                        "type": "function",
                        "function": [
                            "name": "file_write",
                            "arguments": arguments,
                        ] as [String: any Sendable],
                    ] as [String: any Sendable]
                ] as [any Sendable],
            ],
            ["role": "tool", "tool_call_id": "call_1", "content": "{\"ok\":true}"],
        ]
        try expectPrefixRoundTrip(messages, tools: weatherTools(), label: "tool-call history")
    }

    @Test("a transcript ending in an assistant turn appends no generation rail")
    func trailingAssistantTurnAppendsNoRail() throws {
        // Documents the encoder contract that broke boundary derivation. DSV4's
        // official Python encoder emits `<｜Assistant｜>` only after a
        // user/developer/latest_reminder message, so an assistant continuation
        // renders identically with and without the generation prompt. That is
        // CORRECT for the format — it is `canonicalChatCacheBoundaries` that
        // must cope with it (see `trailingContinuationBoundary`).
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a coding agent."],
            ["role": "user", "content": "Build me a minesweeper game"],
            ["role": "assistant", "content": "Working on it.", "reasoning_content": "plan"],
        ]
        let withRail = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: messages, tools: weatherTools(),
            additionalContext: nil, addGenerationPrompt: true)
        let withoutRail = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: messages, tools: weatherTools(),
            additionalContext: nil, addGenerationPrompt: false)

        #expect(withRail == withoutRail, "a continuation turn must not gain a rail")

        // And the transcript WITHOUT that trailing assistant turn is the
        // strictly shorter prefix the boundary fallback relies on.
        let dropped = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: Array(messages.dropLast()), tools: weatherTools(),
            additionalContext: nil, addGenerationPrompt: false)
        #expect(dropped.count < withRail.count)
        #expect(withRail.hasPrefix(dropped))
    }

    @Test("tool-call re-render is byte-stable across processes")
    func toolCallRenderIsDeterministic() throws {
        // `renderToolCallInvoke(name:params:)` used to walk an unordered
        // Dictionary, so chat history re-rendered a tool call with a different
        // parameter order in every process. That changes the prompt bytes, so
        // the prefix-cache boundary for every turn after a tool call stops
        // matching across app restarts. Many keys, so a random order would
        // essentially never coincide.
        let params: [String: Any] = [
            "path": "/tmp/m.html", "content": "<html/>", "mode": "w",
            "encoding": "utf8", "create": true, "retries": 3,
            "owner": "eric", "group": "staff", "backup": false,
        ]
        let renders = Set(
            (0 ..< 12).map { _ in
                DeepseekV4ChatEncoder.renderToolCallInvoke(name: "file_write", params: params)
            })
        #expect(renders.count == 1, "tool-call render is not byte-stable")

        // And the order is the documented one, not merely self-consistent.
        let rendered = try #require(renders.first)
        let order = params.keys.sorted()
        var cursor = rendered.startIndex
        for key in order {
            let needle = "name=\"\(key)\""
            let found = try #require(
                rendered.range(of: needle, range: cursor ..< rendered.endIndex),
                "parameter \(key) missing or out of order")
            cursor = found.upperBound
        }
    }

    @Test("a completed tool round followed by a new user turn round trips")
    func toolRoundThenUserTurnRoundTrips() throws {
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a coding agent."],
            ["role": "user", "content": "Build me a minesweeper game"],
            [
                "role": "assistant",
                "content": "",
                "reasoning_content": "writing the file",
                "tool_calls": [
                    [
                        "id": "call_1",
                        "type": "function",
                        "function": [
                            "name": "file_write",
                            "arguments": "{\"content\":\"<html/>\",\"path\":\"/tmp/m.html\"}",
                        ] as [String: any Sendable],
                    ] as [String: any Sendable]
                ] as [any Sendable],
            ],
            ["role": "tool", "tool_call_id": "call_1", "content": "{\"ok\":true}"],
            ["role": "assistant", "content": "Done.", "reasoning_content": "confirmed"],
            ["role": "user", "content": "Now add a timer."],
        ]
        try expectPrefixRoundTrip(messages, tools: weatherTools(), label: "tool round + user turn")
    }
}
