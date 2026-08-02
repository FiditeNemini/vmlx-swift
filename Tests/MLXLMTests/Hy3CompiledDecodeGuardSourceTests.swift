import Foundation
import Testing

struct Hy3CompiledDecodeGuardSourceTests {
    private static let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()

    @Test("BatchEngine does not promote Hy3/Hunyuan to compiled decode")
    func batchEngineCompileGuardPinsHy3() throws {
        let source = try String(contentsOf: Self.repoRoot.appendingPathComponent(
            "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift"))

        #expect(source.contains("private var compiledDecodeDeniedForModel: Bool"))
        #expect(source.contains("context.configuration.toolCallFormat == .hunyuan"))
        #expect(source.contains("!compiledDecodeDeniedForModel && !soloParameters.enableCompiledDecode"))
        #expect(source.contains("guard !compiledDecodeDeniedForModel else { return }"))
    }

    @Test("TokenIterator direct compiled decode also denies Hy3/Hunyuan")
    func tokenIteratorCompileGuardPinsHy3() throws {
        let source = try String(contentsOf: Self.repoRoot.appendingPathComponent(
            "Libraries/MLXLMCommon/Evaluate.swift"))

        #expect(source.contains("private static func compiledDecodeDenied(for model: any LanguageModel) -> Bool"))
        #expect(source.contains("typeName.contains(\"hy3\") || typeName.contains(\"hunyuan\")"))
        #expect(source.contains("effectiveParameters.enableCompiledDecode && !Self.compiledDecodeDenied(for: model)"))
    }

    @Test("DSV4 keeps model-native compiled kernels and denies generic whole-cache compile")
    func compiledDecodeGuardsPinDSV4() throws {
        let batchSource = try String(contentsOf: Self.repoRoot.appendingPathComponent(
            "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift"))
        let iteratorSource = try String(contentsOf: Self.repoRoot.appendingPathComponent(
            "Libraries/MLXLMCommon/Evaluate.swift"))

        #expect(batchSource.contains("modelTypeName.contains(\"deepseekv4\")"))
        #expect(iteratorSource.contains("typeName.contains(\"deepseekv4\")"))
        #expect(iteratorSource.contains("stateless gate and"))
        #expect(iteratorSource.contains("skip the incompatible whole-cache trace"))
    }
}
