// Copyright © 2025 Apple Inc.

import Foundation

public struct ToolCall: Hashable, Codable, Sendable {
    /// Represents the function details for a tool call
    public struct Function: Hashable, Codable, Sendable {
        /// The name of the function
        public let name: String

        /// The arguments passed to the function
        public let arguments: [String: JSONValue]

        /// Exact JSON-object text observed at the protocol boundary, when
        /// available. Execution always uses ``arguments``; history encoders
        /// may reuse this only after validating that it decodes to the same
        /// values. Keeping it out of `Codable` preserves the public wire shape.
        public let rawArgumentsJSON: String?

        public init(
            name: String,
            arguments: [String: JSONValue],
            rawArgumentsJSON: String? = nil
        ) {
            self.name = name
            self.arguments = arguments
            self.rawArgumentsJSON = rawArgumentsJSON
        }

        public init(
            name: String,
            arguments: [String: any Sendable],
            rawArgumentsJSON: String? = nil
        ) {
            self.name = name
            self.arguments = arguments.mapValues { JSONValue.from($0) }
            self.rawArgumentsJSON = rawArgumentsJSON
        }

        private enum CodingKeys: String, CodingKey {
            case name
            case arguments
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.name = try container.decode(String.self, forKey: .name)
            self.arguments = try container.decode(
                [String: JSONValue].self,
                forKey: .arguments
            )
            self.rawArgumentsJSON = nil
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(name, forKey: .name)
            try container.encode(arguments, forKey: .arguments)
        }
    }

    /// Stable id for correlating a later tool-role result with this call.
    public let id: String?

    /// The function to be called
    public let function: Function

    public init(function: Function, id: String? = nil) {
        self.id = id
        self.function = function
    }

    public init(id: String?, function: Function) {
        self.id = id
        self.function = function
    }
}

extension ToolCall {
    public func execute<Input, Output>(with tool: Tool<Input, Output>) async throws -> Output {
        // Check that the tool name matches the function name
        guard tool.name == function.name else {
            throw ToolError.nameMismatch(toolName: tool.name, functionName: function.name)
        }

        // Convert the JSONValue arguments dictionary to a JSON-encoded Data object
        let jsonObject = function.arguments.mapValues { $0.anyValue }
        let jsonData = try JSONSerialization.data(withJSONObject: jsonObject)

        // Decode the Input type from the JSON data
        let input = try JSONDecoder().decode(Input.self, from: jsonData)

        // Execute the tool's handler with the decoded input
        return try await tool.handler(input)
    }
}

// Define Tool-related errors
public enum ToolError: Error, LocalizedError {
    case nameMismatch(toolName: String, functionName: String)

    public var errorDescription: String? {
        switch self {
        case .nameMismatch(let toolName, let functionName):
            return "Tool name mismatch: expected '\(toolName)' but got '\(functionName)'"
        }
    }
}
