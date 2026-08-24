//
//  TestSourcesAreRegisteredTests.swift
//  MLXLMCommonFocusedTests
//
//  `MLXLMCommonFocusedTests` enumerates its `sources:` explicitly in
//  Package.swift. A test file added to the directory but not to that list
//  compiles NOWHERE, and nothing says so: the build is green, and the suite
//  reports no absence because from its point of view there is none. The file
//  is not skipped, not disabled, not failing — it does not exist.
//
//  This has now happened twice. An earlier revision of #300 shipped
//  `ReasoningEffortPolicyTests.swift` unregistered, so 20 tests had never
//  executed once; registering them surfaced a raw-string escape and two
//  genuine failures that changed a contract. And `NormConventionResolverTests.swift`
//  was added in 36ff6201 and never appeared in Package.swift at all —
//  `git log -S` over Package.swift returns nothing for it.
//
//  A green suite that silently covers less than it appears to is the worst
//  shape a harness can take, so this asserts the invariant directly rather
//  than trusting the next person to remember the list.
//

import Foundation
import Testing

@Suite("Every focused test file is registered in Package.swift")
struct TestSourcesAreRegisteredTests {

    /// Walk up from this file to the package root, so the check does not
    /// depend on the working directory the test runner happens to use.
    private static func packageRoot(from file: StaticString = #filePath) -> URL? {
        var dir = URL(fileURLWithPath: "\(file)").deletingLastPathComponent()
        for _ in 0 ..< 6 {
            if FileManager.default.fileExists(atPath: dir.appendingPathComponent("Package.swift").path) {
                return dir
            }
            dir = dir.deletingLastPathComponent()
        }
        return nil
    }

    @Test("no .swift file in the target directory is missing from sources:")
    func everyFileOnDiskIsListed() throws {
        let root = try #require(Self.packageRoot(), "could not locate Package.swift")
        let manifest = try String(
            contentsOf: root.appendingPathComponent("Package.swift"), encoding: .utf8)
        let targetDir = root
            .appendingPathComponent("Tests")
            .appendingPathComponent("MLXLMCommonFocusedTests")

        let onDisk = try FileManager.default
            .contentsOfDirectory(atPath: targetDir.path)
            .filter { $0.hasSuffix(".swift") }
            .sorted()
        // The directory is the source of truth for what SHOULD run; if it is
        // empty the check has lost its subject and must not silently pass.
        #expect(!onDisk.isEmpty, "found no .swift files — the check is looking in the wrong place")

        // Scope to this target's sources array. Matching the whole manifest
        // would let a filename registered under a DIFFERENT target satisfy
        // the check, which is the same false pass in a new disguise.
        let marker = "\"MLXLMCommonFocusedTests\""
        let targetStart = try #require(
            manifest.range(of: marker), "target declaration not found in Package.swift")
        let afterTarget = manifest[targetStart.upperBound...]
        let sourcesStart = try #require(
            afterTarget.range(of: "sources:"), "target has no explicit sources: list")
        let afterSources = afterTarget[sourcesStart.upperBound...]
        let sourcesEnd = try #require(afterSources.firstIndex(of: "]"), "unterminated sources: list")
        let sourcesBlock = String(afterSources[..<sourcesEnd])

        let unlisted = onDisk.filter { !sourcesBlock.contains("\"\($0)\"") }
        #expect(
            unlisted.isEmpty,
            """
            These test files exist on disk but are NOT in the MLXLMCommonFocusedTests \
            sources: list, so they compile nowhere and have never run: \
            \(unlisted.joined(separator: ", ")). \
            Add each to the sources: array in Package.swift.
            """
        )
    }
}
