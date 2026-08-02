// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLXLMCommon
import Testing

@Suite("DSV4 DSML tool parser focused contracts")
struct DSMLToolCallParserFocusedTests {
    @Test("DSML parser extracts every invoke and preserves typed parameters")
    func parserExtractsEveryInvokeAndTypedParameters() {
        let dsml = DeepseekV4Tokens.dsml
        let output = """
            <\(dsml)tool_calls>
            <\(dsml)invoke name="get_weather">
            <\(dsml)parameter name="city" string="true">Paris</\(dsml)parameter>
            <\(dsml)parameter name="days" string="false">3</\(dsml)parameter>
            </\(dsml)invoke>
            <\(dsml)invoke name="set_alarm">
            <\(dsml)parameter name="enabled" string="false">true</\(dsml)parameter>
            <\(dsml)parameter name="tags" string="false">["morning","work"]</\(dsml)parameter>
            </\(dsml)invoke>
            </\(dsml)tool_calls>
            """

        let calls = DSMLToolCallParser().parseEOS(output, tools: nil)

        #expect(calls.count == 2)
        #expect(calls[0].function.name == "get_weather")
        #expect(calls[0].function.arguments["city"] == .string("Paris"))
        #expect(calls[0].function.arguments["days"] == .int(3))
        #expect(calls[1].function.name == "set_alarm")
        #expect(calls[1].function.arguments["enabled"] == .bool(true))
        #expect(
            calls[1].function.arguments["tags"]
                == .array([.string("morning"), .string("work")])
        )
    }

    @Test("canonical DSML streams incrementally and executes exactly once")
    func canonicalDSMLStreamsIncrementallyAndExecutesExactlyOnce() {
        let dsml = DeepseekV4Tokens.dsml
        let output = """
            <\(dsml)tool_calls>
            <\(dsml)invoke name="file_read">
            <\(dsml)parameter name="path" string="true">mandelbrot.py</\(dsml)parameter>
            <\(dsml)parameter name="start_line" string="false">38</\(dsml)parameter>
            <\(dsml)parameter name="end_line" string="false">41</\(dsml)parameter>
            </\(dsml)invoke>
            </\(dsml)tool_calls>
            """
        let processor = ToolCallProcessor(format: .dsml, tools: fileReadToolSchema())
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "file_read")
        #expect(processor.toolCalls.first?.function.arguments["path"] == .string("mandelbrot.py"))
        #expect(processor.toolCalls.first?.function.arguments["start_line"] == .int(38))
        #expect(processor.toolCalls.first?.function.arguments["end_line"] == .int(41))
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!visible.contains("DSML"))
    }

    @Test("native DSML accepts only canonical tags and never executes alternate syntaxes")
    func nativeDSMLAcceptsOnlyCanonicalTags() {
        let noncanonicalToolText = [
            "_only:request_tool<invoke><target_name>file_read</target><params><string>mandelbrot.py</string></params></invoke>",
            #"file_read("path": "mandelbrot.py")"#,
            #"{"tool":"file_read","path":"mandelbrot.py"}"#,
            "file_read\npath=mandelbrot.py",
        ]

        for (index, body) in noncanonicalToolText.enumerated() {
            #expect(
                DSMLToolCallParser().parse(
                    content: body,
                    tools: strictFileReadToolSchema()) == nil,
                "noncanonical fixture \(index) parsed directly"
            )
            #expect(
                DSMLToolCallParser().parseEOS(
                    body,
                    tools: strictFileReadToolSchema()).isEmpty,
                "noncanonical fixture \(index) parsed at EOS"
            )

            let processor = ToolCallProcessor(
                format: .dsml,
                tools: strictFileReadToolSchema())
            var visible = ""
            for character in body {
                visible += processor.processChunk(String(character)) ?? ""
            }
            visible += processor.processEOS() ?? ""
            #expect(processor.toolCalls.isEmpty, "noncanonical fixture \(index) executed")
            #expect(
                visible == body,
                "noncanonical fixture \(index) was not preserved as non-executable text"
            )

            expectCanonicalRejection(
                canonicalEnvelope(body: body),
                tools: strictFileReadToolSchema(),
                label: "wrapped fallback \(index)"
            )
        }
    }

    @Test("canonical DSML rejects an unregistered invoke")
    func canonicalDSMLRejectsUnregisteredInvoke() {
        let dsml = DeepseekV4Tokens.dsml
        let output = canonicalEnvelope(
            body: """
                <\(dsml)invoke name="not_registered">
                <\(dsml)parameter name="path" string="true">mandelbrot.py</\(dsml)parameter>
                </\(dsml)invoke>
                """)

        expectCanonicalRejection(
            output,
            tools: strictFileReadToolSchema(),
            label: "unknown tool"
        )
    }

    @Test("canonical DSML rejects a missing required argument")
    func canonicalDSMLRejectsMissingRequiredArgument() {
        let dsml = DeepseekV4Tokens.dsml
        let output = canonicalEnvelope(
            body: """
                <\(dsml)invoke name="file_read">
                <\(dsml)parameter name="start_line" string="false">38</\(dsml)parameter>
                </\(dsml)invoke>
                """)

        expectCanonicalRejection(
            output,
            tools: strictFileReadToolSchema(),
            label: "missing required path"
        )
    }

    @Test("canonical DSML rejects an unknown argument when additional properties are disabled")
    func canonicalDSMLRejectsDisallowedUnknownArgument() {
        let dsml = DeepseekV4Tokens.dsml
        let output = canonicalEnvelope(
            body: """
                <\(dsml)invoke name="file_read">
                <\(dsml)parameter name="path" string="true">mandelbrot.py</\(dsml)parameter>
                <\(dsml)parameter name="surprise" string="true">execute-me</\(dsml)parameter>
                </\(dsml)invoke>
                """)

        expectCanonicalRejection(
            output,
            tools: strictFileReadToolSchema(),
            label: "disallowed unknown argument"
        )
    }

    @Test("canonical DSML rejects duplicate parameters")
    func canonicalDSMLRejectsDuplicateParameters() {
        let dsml = DeepseekV4Tokens.dsml
        let output = canonicalEnvelope(
            body: """
                <\(dsml)invoke name="file_read">
                <\(dsml)parameter name="path" string="true">first.py</\(dsml)parameter>
                <\(dsml)parameter name="path" string="true">second.py</\(dsml)parameter>
                </\(dsml)invoke>
                """)

        expectCanonicalRejection(
            output,
            tools: strictFileReadToolSchema(),
            label: "duplicate path"
        )
    }

    @Test("canonical DSML rejects malformed non-string JSON")
    func canonicalDSMLRejectsMalformedNonStringJSON() {
        let dsml = DeepseekV4Tokens.dsml
        let output = canonicalEnvelope(
            body: """
                <\(dsml)invoke name="file_read">
                <\(dsml)parameter name="path" string="true">mandelbrot.py</\(dsml)parameter>
                <\(dsml)parameter name="start_line" string="false">{"broken":]</\(dsml)parameter>
                </\(dsml)invoke>
                """)

        expectCanonicalRejection(
            output,
            tools: strictFileReadToolSchema(),
            label: "malformed non-string JSON"
        )
    }

    @Test("canonical DSML preserves valid nested objects arrays strings and escaping")
    func canonicalDSMLPreservesNestedValuesAndEscaping() {
        let dsml = DeepseekV4Tokens.dsml
        let payload =
            #"{"metadata":{"label":"a\"b","path":"C:\\tmp","enabled":true},"#
            + #""items":[{"name":"alpha","values":[1,2]},{"name":"beta","values":[3]}]}"#
        let note = "quote: \"hello\" \\\\server\\share\nsecond line"
        let output = canonicalEnvelope(
            body: """
                <\(dsml)invoke name="complex_tool">
                <\(dsml)parameter name="payload" string="false">\(payload)</\(dsml)parameter>
                <\(dsml)parameter name="note" string="true">\(note)</\(dsml)parameter>
                </\(dsml)invoke>
                """)

        let calls = DSMLToolCallParser().parseEOS(output, tools: complexToolSchema())

        #expect(calls.count == 1)
        #expect(calls.first?.function.name == "complex_tool")
        #expect(calls.first?.function.arguments["note"] == .string(note))
        #expect(
            calls.first?.function.arguments["payload"]
                == .object([
                    "metadata": .object([
                        "label": .string("a\"b"),
                        "path": .string("C:\\tmp"),
                        "enabled": .bool(true),
                    ]),
                    "items": .array([
                        .object([
                            "name": .string("alpha"),
                            "values": .array([.int(1), .int(2)]),
                        ]),
                        .object([
                            "name": .string("beta"),
                            "values": .array([.int(3)]),
                        ]),
                    ]),
                ])
        )
    }

    @Test("canonical DSML validates and preserves parallel invokes atomically")
    func canonicalDSMLPreservesParallelInvokesAtomically() {
        let dsml = DeepseekV4Tokens.dsml
        let output = canonicalEnvelope(
            body: """
                <\(dsml)invoke name="get_weather">
                <\(dsml)parameter name="location" string="true">Paris</\(dsml)parameter>
                </\(dsml)invoke>
                <\(dsml)invoke name="set_alarm">
                <\(dsml)parameter name="enabled" string="false">true</\(dsml)parameter>
                <\(dsml)parameter name="tags" string="false">["morning","work"]</\(dsml)parameter>
                </\(dsml)invoke>
                """)

        let calls = DSMLToolCallParser().parseEOS(output, tools: parallelToolSchema())

        #expect(calls.count == 2)
        #expect(calls[0].function.name == "get_weather")
        #expect(calls[0].function.arguments["location"] == .string("Paris"))
        #expect(calls[1].function.name == "set_alarm")
        #expect(calls[1].function.arguments["enabled"] == .bool(true))
        #expect(
            calls[1].function.arguments["tags"]
                == .array([.string("morning"), .string("work")])
        )

        let invalidSecond = output.replacingOccurrences(
            of: #"name="enabled" string="false">true"#,
            with: #"name="enabled" string="false">not-json"#
        )
        expectCanonicalRejection(
            invalidSecond,
            tools: parallelToolSchema(),
            label: "invalid second parallel invoke"
        )
    }

    @Test("completed malformed DSML aliases remain non-executable")
    func completedMalformedDSMLAliasesRemainNonExecutable() {
        let dsml = DeepseekV4Tokens.dsml
        let canonicalOpen = "<\(dsml)tool_calls>"
        let invoke = """
            <\(dsml)invoke name="file_read">
            <\(dsml)parameter name="path" string="true">mandelbrot.py</\(dsml)parameter>
            """
        let canonicalInvokeClose = "</\(dsml)invoke>"
        let canonicalOuterClose = "</\(dsml)tool_calls>"
        let malformed: [String] = [
            "\(canonicalOpen)\n\(invoke)\n</\(dsml)inv>\n\(canonicalOuterClose)",
            "\(canonicalOpen)\n\(invoke)\n</inv>\n\(canonicalOuterClose)",
            "\(canonicalOpen)\n\(invoke)\n\(canonicalInvokeClose)\n</\(dsml)tool_cs>",
            "\(canonicalOpen)\n\(invoke)\n\(canonicalInvokeClose)\n</tool_cs>",
            "<\(dsml)tool_ccalls>\n\(invoke)\n</\(dsml)inv>\n</\(dsml)tool_cs>",
            "<\(dsml)tool_cimport>\n\(invoke)\n\(canonicalInvokeClose)\n</\(dsml)tool_cimport>",
        ]

        for (index, output) in malformed.enumerated() {
            #expect(
                DSMLToolCallParser().parseEOS(output, tools: fileReadToolSchema()).isEmpty,
                "malformed fixture \(index) parsed directly"
            )

            let processor = ToolCallProcessor(format: .dsml, tools: fileReadToolSchema())
            var visible = ""
            for ch in output {
                visible += processor.processChunk(String(ch)) ?? ""
            }
            visible += processor.processEOS() ?? ""

            #expect(processor.toolCalls.isEmpty, "malformed fixture \(index) executed")
            #expect(
                visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                "malformed fixture \(index) leaked protocol: \(visible)"
            )
        }
    }

    @Test("DSV4 processor does not execute request_tool XML")
    func processorRejectsRequestToolXMLRail() {
        let output =
            "_only:request_tool<invoke><target_name>line_count</target><params><string>\n"
            + #"one\ntwo"#
            + "\n</string></params></invoke>"
        let processor = ToolCallProcessor(format: .dsml, tools: lineCountToolSchema())
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.isEmpty)
        #expect(visible == output)
    }

    @Test("DSML processor does not execute bare-name malformed JSON")
    func processorRejectsBareNameJSONWithMalformedStringEscape() {
        let output = #"line_count{"text":"alpha\nbeta\gamma"}"#
        let processor = ToolCallProcessor(format: .dsml, tools: lineCountToolSchema())
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.isEmpty)
        #expect(visible == output)
    }

    @Test("DSML processor suppresses incomplete protocol opener at EOS")
    func processorSuppressesIncompleteProtocolOpenerAtEOS() {
        let dsml = DeepseekV4Tokens.dsml
        let output = "\n\n<\(dsml)tool_c"
        let processor = ToolCallProcessor(format: .dsml, tools: fileReadToolSchema())
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.isEmpty)
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!visible.contains("DSML"))
        #expect(!visible.contains("tool_c"))
    }

    @Test("DSML processor routes Osaurus folder and git tools through canonical streaming")
    func processorRoutesOsaurusFolderAndGitToolsThroughCanonicalStreaming() {
        let fixtures: [DSMLToolFixture] = [
            .init(
                name: "file_tree",
                parameters: [
                    .init(name: "path", value: ".", string: true, expected: .string(".")),
                    .init(name: "max_depth", value: "2", string: false, expected: .int(2)),
                ]
            ),
            .init(
                name: "file_read",
                parameters: [
                    .init(name: "path", value: "mandelbrot.py", string: true, expected: .string("mandelbrot.py")),
                    .init(name: "start_line", value: "38", string: false, expected: .int(38)),
                    .init(name: "end_line", value: "41", string: false, expected: .int(41)),
                ]
            ),
            .init(
                name: "file_write",
                parameters: [
                    .init(name: "path", value: "osaurus_probe.txt", string: true, expected: .string("osaurus_probe.txt")),
                    .init(name: "content", value: "alpha\nbeta", string: true, expected: .string("alpha\nbeta")),
                ]
            ),
            .init(
                name: "file_edit",
                parameters: [
                    .init(name: "path", value: "osaurus_probe.txt", string: true, expected: .string("osaurus_probe.txt")),
                    .init(name: "old_string", value: "alpha", string: true, expected: .string("alpha")),
                    .init(name: "new_string", value: "beta", string: true, expected: .string("beta")),
                ]
            ),
            .init(
                name: "file_search",
                parameters: [
                    .init(name: "pattern", value: "np.clip", string: true, expected: .string("np.clip")),
                    .init(name: "path", value: "mandelbrot.py", string: true, expected: .string("mandelbrot.py")),
                    .init(name: "max_results", value: "3", string: false, expected: .int(3)),
                ]
            ),
            .init(
                name: "shell_run",
                parameters: [
                    .init(name: "command", value: "printf ok", string: true, expected: .string("printf ok")),
                    .init(name: "timeout", value: "5", string: false, expected: .int(5)),
                ]
            ),
            .init(name: "git_status", parameters: []),
            .init(
                name: "git_diff",
                parameters: [
                    .init(name: "path", value: "mandelbrot.py", string: true, expected: .string("mandelbrot.py")),
                    .init(name: "staged", value: "false", string: false, expected: .bool(false)),
                ]
            ),
            .init(
                name: "git_commit",
                parameters: [
                    .init(name: "message", value: "probe commit", string: true, expected: .string("probe commit")),
                ]
            ),
        ]

        for fixture in fixtures {
            let processor = ToolCallProcessor(format: .dsml)
            var visible = ""
            for ch in canonicalDSML(for: fixture) {
                visible += processor.processChunk(String(ch)) ?? ""
            }
            visible += processor.processEOS() ?? ""

            #expect(
                visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                "\(fixture.name) DSML leaked visible text: \(visible)"
            )
            #expect(!visible.contains("DSML"), "\(fixture.name) leaked DSML marker: \(visible)")
            #expect(processor.toolCalls.count == 1, "\(fixture.name) should emit one tool call")

            let call = processor.toolCalls.first
            #expect(call?.function.name == fixture.name)
            for parameter in fixture.parameters {
                assertArgument(
                    call?.function.arguments[parameter.name],
                    matches: parameter.expected,
                    tool: fixture.name,
                    parameter: parameter.name
                )
            }
        }
    }

    @Test("DSML does not execute schema-valid inline JSON")
    func inlineJSONIsNotExecutableDSML() {
        let output = """
            {"tool":"file_read","path":"mandelbrot.py","start_line":38,"end_line":41}
            """
        let processor = ToolCallProcessor(format: .dsml, tools: fileReadToolSchema())
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.isEmpty)
        #expect(visible == output)
    }

    @Test("DSML does not execute malformed tool-shaped inline JSON")
    func malformedInlineJSONIsNotExecutableDSML() {
        let output = """
            {"tool":"file_read","r":"np.clip(esc * 4.0 - 1.0, 0.0, 1.0)","g":"np.clip(1.0 - np.abs(esc * 2.0 - 1.0), 0.0, 1.0)","b":"np.clip(1.0 - esc * 2.0, 0.0, 1.0)"}
            """
        let processor = ToolCallProcessor(format: .dsml, tools: fileReadToolSchema())
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.isEmpty)
        #expect(visible == output)
    }

    @Test("DSML tool-like reasoning content is never executable")
    func toolLikeReasoningContentIsNeverExecutable() {
        let dsml = DeepseekV4Tokens.dsml
        let fixtures = [
            canonicalEnvelope(
                body: """
                    <\(dsml)invoke name="file_read">
                    <\(dsml)parameter name="path" string="true">mandelbrot.py</\(dsml)parameter>
                    </\(dsml)invoke>
                    """),
            #"{"tool":"file_read","path":"mandelbrot.py"}"#,
            #"file_read("path": "mandelbrot.py")"#,
            "file_read\npath=mandelbrot.py",
        ]

        #expect(ToolCallFormat.dsml.parsesToolCallsFromReasoningChannel == false)
        #expect(ToolCallFormat.json.parsesToolCallsFromReasoningChannel == true)

        for (index, text) in fixtures.enumerated() {
            let processor = ToolCallProcessor(
                format: .dsml,
                tools: strictFileReadToolSchema())
            let events = routeGenerationText(
                text,
                channel: .reasoning,
                through: processor)

            #expect(processor.toolCalls.isEmpty, "reasoning fixture \(index) executed")
            #expect(events.compactMap(\.toolCall).isEmpty)
            #expect(events.compactMap(\.reasoning).joined() == text)
            #expect(events.compactMap(\.chunk).isEmpty)
        }
    }

    @Test("DSV4 instruct prompt routes DSML output to tool calls without reasoning leakage")
    func instructPromptRoutesDSMLWithoutReasoningLeakage() {
        let prompt = DeepseekV4ChatEncoder().encode(
            messages: [.init(role: .user, content: "Weather in Paris?")],
            thinkingMode: .chat
        )
        #expect(prompt.hasSuffix(DeepseekV4Tokens.thinkEnd))

        var reasoningParser = ReasoningParser.forPrompt(
            stampName: "think_xml",
            promptTail: promptTail(prompt)
        )
        let toolProcessor = ToolCallProcessor(format: .dsml)
        let dsml = DeepseekV4Tokens.dsml
        let output = """
            <\(dsml)tool_calls>
            <\(dsml)invoke name="get_weather">
            <\(dsml)parameter name="city" string="true">Paris</\(dsml)parameter>
            </\(dsml)invoke>
            </\(dsml)tool_calls>
            """

        var reasoning = ""
        var visible = ""
        for ch in output {
            if var parser = reasoningParser {
                for segment in parser.feed(String(ch)) {
                    switch segment {
                    case .reasoning(let text):
                        reasoning += text
                    case .content(let text):
                        visible += toolProcessor.processChunk(text) ?? ""
                    }
                }
                reasoningParser = parser
            } else {
                visible += toolProcessor.processChunk(String(ch)) ?? ""
            }
        }
        if var parser = reasoningParser {
            for segment in parser.flush() {
                switch segment {
                case .reasoning(let text):
                    reasoning += text
                case .content(let text):
                    visible += toolProcessor.processChunk(text) ?? ""
                }
            }
            reasoningParser = parser
        }
        toolProcessor.processEOS()

        #expect(reasoning.isEmpty)
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(toolProcessor.toolCalls.count == 1)
        #expect(toolProcessor.toolCalls.first?.function.name == "get_weather")
        #expect(toolProcessor.toolCalls.first?.function.arguments["city"] == .string("Paris"))
    }

    @Test("DSV4 chunked reasoning close routes following DSML to tool call")
    func chunkedReasoningCloseRoutesFollowingDSMLToToolCall() {
        var reasoningParser = ReasoningParser.forPrompt(
            stampName: "think_xml",
            promptTail: "<\u{FF5C}Assistant\u{FF5C}><think>"
        )
        let toolProcessor = ToolCallProcessor(format: .dsml)
        let dsml = DeepseekV4Tokens.dsml
        let chunks = [
            "Need the weather</think>",
            """
            <\(dsml)tool_calls>
            <\(dsml)invoke name="get_weather">
            <\(dsml)parameter name="location" string="true">Paris</\(dsml)parameter>
            </\(dsml)invoke>
            </\(dsml)tool_calls>
            """,
        ]

        var reasoning = ""
        var visible = ""
        var calls: [ToolCall] = []
        func route(_ segments: [ReasoningSegment]) {
            for segment in segments {
                let events: [Generation]
                switch segment {
                case .reasoning(let text):
                    events = routeGenerationText(
                        text, channel: .reasoning, through: toolProcessor)
                case .content(let text):
                    events = routeGenerationText(text, channel: .content, through: toolProcessor)
                }
                for event in events {
                    switch event {
                    case .reasoning(let text): reasoning += text
                    case .chunk(let text): visible += text
                    case .toolCall(let call): calls.append(call)
                    default: break
                    }
                }
            }
        }

        for chunk in chunks {
            if var parser = reasoningParser {
                route(parser.feed(chunk))
                reasoningParser = parser
            }
        }
        if var parser = reasoningParser {
            route(parser.flush())
            reasoningParser = parser
        }
        for event in flushGenerationText(channel: .content, through: toolProcessor) {
            switch event {
            case .chunk(let text): visible += text
            case .toolCall(let call): calls.append(call)
            default: break
            }
        }

        #expect(reasoning == "Need the weather")
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(calls.count == 1)
        #expect(calls.first?.function.name == "get_weather")
        #expect(calls.first?.function.arguments["location"] == .string("Paris"))
    }

    @Test("DSV4 Osaurus split reasoning and DSML stream routes to one tool call")
    func osaurusSplitReasoningAndDSMLStreamRoutesToOneToolCall() {
        var reasoningParser = ReasoningParser.forPrompt(
            stampName: "think_xml",
            promptTail: "<\u{FF5C}Assistant\u{FF5C}><think>"
        )!
        let toolCallProcessor = ToolCallProcessor(format: .dsml)
        var events: [Generation] = []

        func route(_ text: String, channel: GenerationTextChannel) {
            events.append(
                contentsOf: routeGenerationText(
                    text,
                    channel: channel,
                    through: toolCallProcessor
                )
            )
        }

        for raw in [
            "Need the weather</think>",
            "<\u{FF5C}DSML\u{FF5C}tool_calls>\n",
            "<\u{FF5C}DSML\u{FF5C}invoke name=\"get_weather\">\n",
            "<\u{FF5C}DSML\u{FF5C}parameter name=\"location\" string=\"true\">Paris</\u{FF5C}DSML\u{FF5C}parameter>\n",
            "</\u{FF5C}DSML\u{FF5C}invoke>\n",
            "</\u{FF5C}DSML\u{FF5C}tool_calls>",
        ] {
            for segment in reasoningParser.feed(raw) {
                switch segment {
                case .reasoning(let reasoning):
                    route(reasoning, channel: .reasoning)
                case .content(let content):
                    route(content, channel: .content)
                }
            }
        }
        for segment in reasoningParser.flush() {
            switch segment {
            case .reasoning(let reasoning):
                route(reasoning, channel: .reasoning)
            case .content(let content):
                route(content, channel: .content)
            }
        }
        if let visible = toolCallProcessor.processEOS() {
            route(visible, channel: .content)
        }
        events.append(contentsOf: drainToolCallEvents(from: toolCallProcessor))

        let reasoning = events.compactMap(\.reasoning).joined()
        let visible = events.compactMap(\.chunk).joined()
        let calls = events.compactMap(\.toolCall)

        #expect(reasoning == "Need the weather")
        #expect(visible.isEmpty)
        #expect(calls.count == 1)
        #expect(calls.first?.function.name == "get_weather")
        #expect(calls.first?.function.arguments["location"] == .string("Paris"))
    }

    @Test("DSV4 split DSML chunks buffer directly in the tool processor")
    func splitDSMLChunksBufferDirectlyInToolProcessor() {
        let processor = ToolCallProcessor(format: .dsml)
        let chunks = [
            "<\u{FF5C}DSML\u{FF5C}tool_calls>\n",
            "<\u{FF5C}DSML\u{FF5C}invoke name=\"get_weather\">\n",
            "<\u{FF5C}DSML\u{FF5C}parameter name=\"location\" string=\"true\">Paris</\u{FF5C}DSML\u{FF5C}parameter>\n",
            "</\u{FF5C}DSML\u{FF5C}invoke>\n",
            "</\u{FF5C}DSML\u{FF5C}tool_calls>",
        ]

        var visible = ""
        for chunk in chunks {
            visible += processor.processChunk(chunk) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(visible.isEmpty)
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "get_weather")
        #expect(processor.toolCalls.first?.function.arguments["location"] == .string("Paris"))
    }

    @Test("DSV4 ReasoningParser split DSML chunks buffer directly in the tool processor")
    func reasoningParserSplitDSMLChunksBufferDirectlyInToolProcessor() {
        let processor = ToolCallProcessor(format: .dsml)
        let chunks = [
            "<\u{FF5C}DSML\u{FF5C}tool_",
            "calls>\n<\u{FF5C}DSML\u{FF5C}invoke name=\"get_wea",
            "ther\">\n<\u{FF5C}DSML\u{FF5C}parameter name=\"location\" string=\"true\">Paris</\u{FF5C}DSML\u{FF5C}para",
            "meter>\n</\u{FF5C}DSML\u{FF5C}i",
            "nvoke>\n</\u{FF5C}DSML\u{FF5C}tool",
            "_calls>",
        ]

        var visible = ""
        for chunk in chunks {
            visible += processor.processChunk(chunk) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(visible.isEmpty)
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "get_weather")
        #expect(processor.toolCalls.first?.function.arguments["location"] == .string("Paris"))
    }

    @Test("DSV4 capability aliases route to DSML before generic DeepSeek")
    func capabilityAliasesPreferDSML() {
        for stamp in ["dsml", "deepseek_v4", "deepseek_v4_flash", "deepseekv4"] {
            #expect(ToolCallFormat.fromCapabilityName(stamp) == .dsml)
        }
        #expect(ToolCallFormat.fromCapabilityName("deepseek") == .glm4)
        #expect(ToolCallFormat.fromCapabilityName("deepseek_v3") == .glm4)
    }

    private func promptTail(_ prompt: String) -> String {
        let start =
            prompt.index(
                prompt.endIndex,
                offsetBy: -256,
                limitedBy: prompt.startIndex
            ) ?? prompt.startIndex
        return String(prompt[start...])
    }

    private func canonicalEnvelope(body: String) -> String {
        let dsml = DeepseekV4Tokens.dsml
        return """
            <\(dsml)tool_calls>
            \(body)
            </\(dsml)tool_calls>
            """
    }

    private func expectCanonicalRejection(
        _ output: String,
        tools: [[String: any Sendable]],
        label: String
    ) {
        #expect(
            DSMLToolCallParser().parseEOS(output, tools: tools).isEmpty,
            "\(label) parsed directly"
        )

        let processor = ToolCallProcessor(format: .dsml, tools: tools)
        var visible = ""
        for character in output {
            visible += processor.processChunk(String(character)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.isEmpty, "\(label) became executable")
        #expect(
            processor.toolCallProtocolFailure == .malformedEnvelope,
            "\(label) did not surface the typed protocol failure"
        )
        #expect(
            visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "\(label) leaked protocol bytes: \(visible)"
        )
    }

    private struct DSMLToolFixture {
        let name: String
        let parameters: [DSMLParameterFixture]
    }

    private struct DSMLParameterFixture {
        let name: String
        let value: String
        let string: Bool
        let expected: DSMLExpectedArgument
    }

    private enum DSMLExpectedArgument {
        case string(String)
        case int(Int)
        case bool(Bool)
    }

    private func canonicalDSML(for fixture: DSMLToolFixture) -> String {
        let dsml = DeepseekV4Tokens.dsml
        var lines = [
            "<\(dsml)tool_calls>",
            "<\(dsml)invoke name=\"\(fixture.name)\">",
        ]
        lines += fixture.parameters.map { parameter in
            "<\(dsml)parameter name=\"\(parameter.name)\" string=\"\(parameter.string ? "true" : "false")\">\(parameter.value)</\(dsml)parameter>"
        }
        lines += [
            "</\(dsml)invoke>",
            "</\(dsml)tool_calls>",
        ]
        return lines.joined(separator: "\n")
    }

    private func assertArgument(
        _ actual: (any Sendable)?,
        matches expected: DSMLExpectedArgument,
        tool: String,
        parameter: String
    ) {
        switch expected {
        case .string(let value):
            #expect(actual as? JSONValue == .string(value), "\(tool).\(parameter) mismatch")
        case .int(let value):
            #expect(actual as? JSONValue == .int(value), "\(tool).\(parameter) mismatch")
        case .bool(let value):
            #expect(actual as? JSONValue == .bool(value), "\(tool).\(parameter) mismatch")
        }
    }

    private func fileReadToolSchema() -> [[String: any Sendable]] {
        [
            [
                "type": "function",
                "function": [
                    "name": "file_read",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "path": ["type": "string"] as [String: any Sendable],
                            "start_line": ["type": "integer"] as [String: any Sendable],
                            "end_line": ["type": "integer"] as [String: any Sendable],
                        ] as [String: any Sendable],
                        "required": ["path"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    private func strictFileReadToolSchema() -> [[String: any Sendable]] {
        [
            [
                "type": "function",
                "function": [
                    "name": "file_read",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "path": ["type": "string"] as [String: any Sendable],
                            "start_line": ["type": "integer"] as [String: any Sendable],
                            "end_line": ["type": "integer"] as [String: any Sendable],
                        ] as [String: any Sendable],
                        "required": ["path"],
                        "additionalProperties": false,
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    private func complexToolSchema() -> [[String: any Sendable]] {
        let itemSchema: [String: any Sendable] = [
            "type": "object",
            "properties": [
                "name": ["type": "string"] as [String: any Sendable],
                "values": [
                    "type": "array",
                    "items": ["type": "integer"] as [String: any Sendable]
                ] as [String: any Sendable],
            ] as [String: any Sendable],
            "required": ["name", "values"],
            "additionalProperties": false,
        ]
        let payloadSchema: [String: any Sendable] = [
            "type": "object",
            "properties": [
                "metadata": [
                    "type": "object",
                    "properties": [
                        "label": ["type": "string"] as [String: any Sendable],
                        "path": ["type": "string"] as [String: any Sendable],
                        "enabled": ["type": "boolean"] as [String: any Sendable]
                    ] as [String: any Sendable],
                    "required": ["label", "path", "enabled"],
                    "additionalProperties": false,
                ] as [String: any Sendable],
                "items": [
                    "type": "array",
                    "items": itemSchema
                ] as [String: any Sendable],
            ] as [String: any Sendable],
            "required": ["metadata", "items"],
            "additionalProperties": false,
        ]
        return [
            [
                "type": "function",
                "function": [
                    "name": "complex_tool",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "payload": payloadSchema,
                            "note": ["type": "string"] as [String: any Sendable],
                        ] as [String: any Sendable],
                        "required": ["payload", "note"],
                        "additionalProperties": false,
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    private func parallelToolSchema() -> [[String: any Sendable]] {
        [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "location": ["type": "string"] as [String: any Sendable]
                        ] as [String: any Sendable],
                        "required": ["location"],
                        "additionalProperties": false,
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
            [
                "type": "function",
                "function": [
                    "name": "set_alarm",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "enabled": ["type": "boolean"] as [String: any Sendable],
                            "tags": [
                                "type": "array",
                                "items": ["type": "string"] as [String: any Sendable]
                            ] as [String: any Sendable],
                        ] as [String: any Sendable],
                        "required": ["enabled", "tags"],
                        "additionalProperties": false,
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    private func lineCountToolSchema() -> [[String: any Sendable]] {
        [
            [
                "type": "function",
                "function": [
                    "name": "line_count",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "text": ["type": "string"] as [String: any Sendable],
                        ] as [String: any Sendable],
                        "required": ["text"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

}
