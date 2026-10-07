import Foundation
import XCTest
@testable import MLXLMCommon

/// CPU-only compiler/matcher tests. No model, MLX array or Metal execution.
final class JSONSchemaGrammarTests: XCTestCase {
    private let pieces = ["<eos>", "{", "}", "\"", "answer", ":", "true", "false", " ", "[", "]", ",", "1", "2", "null", "extra", "<end>"]
    private let schema = ##"{"type":"object","properties":{"answer":{"type":"boolean"}},"required":["answer"],"additionalProperties":false}"##

    private func tokenizer() throws -> JSONSchemaGrammarTokenizer {
        try JSONSchemaGrammarTokenizer(vocabulary: pieces, vocabularyType: .raw, stopTokenIDs: [0, 16])
    }
    private func allows(_ mask: JSONSchemaTokenMask, _ token: Int) -> Bool {
        (mask.words[token / 32] & (UInt32(1) << UInt32(token % 32))) != 0
    }
    private func feed(_ grammar: JSONSchemaGrammar, _ tokens: [Int]) throws {
        for token in tokens {
            XCTAssertTrue(allows(try grammar.nextTokenMask(), token), "Expected token \(token) admitted")
            try grammar.accept(tokenID: token)
        }
    }

    func testObjectMaskAndExplicitStopIDs() throws {
        let grammar = try JSONSchemaGrammar(tokenizer: tokenizer(), schema: schema)
        let initial = try grammar.nextTokenMask()
        XCTAssertEqual(initial.vocabularySize, pieces.count)
        XCTAssertFalse(allows(initial, 0))
        XCTAssertFalse(allows(initial, 16))
        XCTAssertThrowsError(try grammar.accept(tokenID: 7))
        try feed(grammar, [1, 3, 4, 3, 5, 6, 2])
        XCTAssertFalse(try grammar.isTerminated())
        let finished = try grammar.nextTokenMask()
        XCTAssertTrue(allows(finished, 0))
        XCTAssertTrue(allows(finished, 16))
        try grammar.accept(tokenID: 16)
        XCTAssertTrue(try grammar.isTerminated())
    }

    func testIndependentCopyReplaysWithoutMutatingOriginal() throws {
        let grammar = try JSONSchemaGrammar(tokenizer: tokenizer(), schema: schema)
        try feed(grammar, [1, 3, 4, 3, 5])
        let originalMask = try grammar.nextTokenMask()
        let copy = try grammar.independentCopy()
        try feed(copy, [7, 2, 0])
        XCTAssertTrue(try copy.isTerminated())
        XCTAssertFalse(try grammar.isTerminated())
        XCTAssertEqual(try grammar.nextTokenMask().words, originalMask.words)
        try feed(grammar, [6, 2, 16])
        XCTAssertTrue(try grammar.isTerminated())
        XCTAssertTrue(try grammar.independentCopy().isTerminated())
    }

    func testOmittedAdditionalPropertiesUsesJSONSchemaDefault() throws {
        let grammar = try JSONSchemaGrammar(tokenizer: tokenizer(), schema: ##"{"type":"object"}"##)
        try feed(grammar, [1, 3, 15, 3, 5, 6, 2, 0])
        XCTAssertTrue(try grammar.isTerminated())
    }

    func testSupportedNestedSchemasAndReferences() throws {
        for schema in [
            ##"{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":2}"##,
            ##"{"type":["string","null"],"enum":["x",null]}"##,
            ##"{"anyOf":[{"type":"boolean"},{"type":"null"}]}"##,
            ##"{"$defs":{"flag":{"type":"boolean"}},"$ref":"#/$defs/flag"}"##,
            ##"{"type":"object","properties":{"not":{"type":"string"}},"required":["not"]}"##,
        ] {
            try JSONSchemaGrammar.validateSupportedSchema(schema)
            _ = try JSONSchemaGrammar(tokenizer: tokenizer(), schema: schema)
        }
    }

    func testUnsupportedKeywordsAndCombinationsFailClosed() {
        let cases = [
            ##"{"oneOf":[{"type":"integer"},{"type":"number"}]}"##,
            ##"{"allOf":[{"type":"string"},{"minLength":3}]}"##,
            ##"{"type":"integer","multipleOf":2}"##,
            ##"{"type":"array","uniqueItems":true}"##,
            ##"{"type":"array","contains":{"type":"integer"}}"##,
            ##"{"type":"object","dependentRequired":{"a":["b"]}}"##,
            ##"{"type":"string","pattern":"a+","minLength":2}"##,
            ##"{"type":"string","format":"email"}"##,
            ##"{"type":"object","properties":{"answer":{"type":"integer","minimum":0}}}"##,
            ##"{"if":{"type":"string"},"then":{"const":"x"}}"##,
            ##"{"$ref":"https://example.invalid/schema"}"##,
            ##"{"$ref":"#/$defs/a~1b","$defs":{"a/b":{"type":"boolean"}}}"##,
            ##"{"enum":["x"],"minLength":2}"##,
            ##"{"anyOf":[{"type":"string"}],"type":"integer"}"##,
        ]
        for schema in cases {
            XCTAssertThrowsError(try JSONSchemaGrammar.validateSupportedSchema(schema), schema)
        }
    }

    func testStringLengthConstraintsAreTypedUnsupported() {
        for keyword in ["minLength", "maxLength"] {
            let schema = "{\"type\":\"string\",\"\(keyword)\":1}"
            XCTAssertThrowsError(try JSONSchemaGrammar.validateSupportedSchema(schema)) { error in
                guard case JSONSchemaGrammarError.unsupportedSchema(let path, let rejected) = error else {
                    return XCTFail("Expected typed unsupported schema error, got \(error)")
                }
                XCTAssertEqual(path, "#")
                XCTAssertEqual(rejected, keyword)
            }
        }
    }

    func testDialectAndReferenceTargetsFailClosed() throws {
        let rejected = [
            ##"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"string"}"##,
            ##"{"$schema":"unknown","type":"string"}"##,
            ##"{"$ref":"#"}"##,
            ##"{"$defs":{"a":{"$ref":"#/$defs/b"},"b":{"$ref":"#/$defs/a"}},"$ref":"#/$defs/a"}"##,
            ##"{"type":"object","properties":{"child":{"$ref":"#"}}}"##,
            ##"{"default":{"type":"string"},"$ref":"#/default"}"##,
            ##"{"examples":[{"type":"string"}],"$ref":"#/examples/0"}"##,
            ##"{"$defs":{"a":{"$ref":"#/$defs/missing"}},"$ref":"#/$defs/a"}"##,
        ]
        for schema in rejected {
            XCTAssertThrowsError(try JSONSchemaGrammar.validateSupportedSchema(schema)) { error in
                guard case JSONSchemaGrammarError.unsupportedSchema = error else {
                    return XCTFail("Expected unsupported schema, got \(error)")
                }
            }
        }
        try JSONSchemaGrammar.validateSupportedSchema(
            ##"{"$defs":{"a":{"$ref":"#/$defs/b"},"b":{"type":"boolean"}},"$ref":"#/$defs/a"}"##)
    }

    func testMalformedSchemasAndTokenizerFailClosed() {
        for schema in ["{", "[]", "42", "false", ##"{"type":42}"##,
            ##"{"type":"mystery"}"##, ##"{"type":"array","minItems":true}"##,
            ##"{"type":"array","minItems":3,"maxItems":2}"##,
            ##"{"type":"string","enum":[true]}"##, ##"{"$ref":"#/$defs/missing"}"##] {
            XCTAssertThrowsError(try JSONSchemaGrammar.validateSupportedSchema(schema), schema)
        }
        XCTAssertThrowsError(try JSONSchemaGrammarTokenizer(vocabulary: pieces, vocabularyType: .raw, stopTokenIDs: []))
        XCTAssertThrowsError(try JSONSchemaGrammarTokenizer(vocabulary: pieces, vocabularyType: .raw, stopTokenIDs: [-1]))
        XCTAssertThrowsError(try JSONSchemaGrammarTokenizer(vocabulary: ["\u{0}"], vocabularyType: .raw, stopTokenIDs: [0]))
    }
}
