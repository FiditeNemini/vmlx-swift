import Foundation

import struct VMLXHub.Config

/// Foundation's ICU-backed regular-expression entry points do not treat a
/// literal JSON-decoded CR/LF inside a character class the same way as the
/// Rust `regex` engine used by Hugging Face tokenizers. Tokenizer JSON stores
/// patterns such as `[\r\n]*`; JSON decoding turns those escapes into actual
/// control scalars. With the scalars passed through unchanged Foundation
/// matches `.` and the following newlines as separate pre-tokens, while the
/// canonical tokenizer keeps `.\n\n` together and permits its learned BPE
/// merge (for DSV4, token `.ĊĊ`).
///
/// Re-escape literal control scalars before compiling/matching a tokenizer
/// regex. This preserves the regex's meaning and matches the representation
/// Foundation handles canonically. Existing textual escapes (`\\n`, `\\r`,
/// and so on) are left untouched.
func foundationCompatibleTokenizerRegex(_ pattern: String) -> String {
    var result = ""
    result.reserveCapacity(pattern.utf8.count)
    for scalar in pattern.unicodeScalars {
        switch scalar.value {
        case 0x08: result += #"\x08"#
        case 0x09: result += #"\t"#
        case 0x0A: result += #"\n"#
        case 0x0C: result += #"\f"#
        case 0x0D: result += #"\r"#
        default: result.unicodeScalars.append(scalar)
        }
    }
    return result
}

enum StringSplitPattern {
    case regexp(regexp: String)
    case string(pattern: String)

    func split(_ text: String, invert: Bool = true) -> [String] {
        switch self {
        case let .regexp(regexp):
            text.split(by: regexp, includeSeparators: true)
        case let .string(substring):
            text.split(by: substring, options: [], includeSeparators: !invert)
        }
    }

    static func from(config: Config) -> StringSplitPattern? {
        if let pattern = config.pattern.String.string() {
            return .string(pattern: pattern)
        }
        if let pattern = config.pattern.Regex.string() {
            return .regexp(regexp: foundationCompatibleTokenizerRegex(pattern))
        }
        return nil
    }
}

enum SplitDelimiterBehavior {
    case removed
    case isolated
    case mergedWithPrevious
    case mergedWithNext
}

extension String {
    func ranges(of string: String, options: CompareOptions = .regularExpression) -> [Range<Index>] {
        var result: [Range<Index>] = []
        var start = startIndex
        while let range = range(of: string, options: options, range: start..<endIndex) {
            result.append(range)
            start = range.lowerBound < range.upperBound ? range.upperBound : index(range.lowerBound, offsetBy: 1, limitedBy: endIndex) ?? endIndex
        }
        return result
    }

    func split(by string: String, options: CompareOptions = .regularExpression, includeSeparators: Bool = false, omittingEmptySubsequences: Bool = true) -> [String] {
        var result: [String] = []
        var start = startIndex
        while let range = range(of: string, options: options, range: start..<endIndex) {
            // Prevent empty strings
            if omittingEmptySubsequences, start < range.lowerBound {
                result.append(String(self[start..<range.lowerBound]))
            }
            if includeSeparators {
                result.append(String(self[range]))
            }
            start = range.upperBound
        }

        if omittingEmptySubsequences, start < endIndex {
            result.append(String(self[start...]))
        }
        return result
    }

    /// This version supports capture groups, wheres the one above doesn't
    func split(by captureRegex: NSRegularExpression) -> [String] {
        // Find the matching capture groups
        let selfRange = NSRange(startIndex..<endIndex, in: self)
        let matches = captureRegex.matches(in: self, options: [], range: selfRange)

        if matches.isEmpty { return [self] }

        var result: [String] = []
        var start = startIndex

        for match in matches {
            // IMPORTANT: convert from NSRange to Range<String.Index>
            // https://stackoverflow.com/questions/75543272/convert-a-given-utf8-nsrange-in-a-string-to-a-utf16-nsrange
            guard let matchRange = Range(match.range, in: self) else { continue }

            // Add text before the match
            if start < matchRange.lowerBound {
                result.append(String(self[start..<matchRange.lowerBound]))
            }

            // Move start to after the match
            start = matchRange.upperBound

            // Append separator, supporting capture groups
            for r in (0..<match.numberOfRanges).reversed() {
                let nsRange = match.range(at: r)
                if let sepRange = Range(nsRange, in: self) {
                    result.append(String(self[sepRange]))
                    break
                }
            }
        }

        // Append remaining suffix
        if start < endIndex {
            result.append(String(self[start...]))
        }

        return result
    }

    func split(by string: String, options: CompareOptions = .regularExpression, behavior: SplitDelimiterBehavior) -> [String] {
        func mergedWithNext(ranges: [Range<String.Index>]) -> [Range<String.Index>] {
            var merged: [Range<String.Index>] = []
            var currentStart = startIndex
            for range in ranges {
                if range.lowerBound == startIndex { continue }
                let mergedRange = currentStart..<range.lowerBound
                currentStart = range.lowerBound
                merged.append(mergedRange)
            }
            if currentStart < endIndex {
                merged.append(currentStart..<endIndex)
            }
            return merged
        }

        func mergedWithPrevious(ranges: [Range<String.Index>]) -> [Range<String.Index>] {
            var merged: [Range<String.Index>] = []
            var currentStart = startIndex
            for range in ranges {
                let mergedRange = currentStart..<range.upperBound
                currentStart = range.upperBound
                merged.append(mergedRange)
            }
            if currentStart < endIndex {
                merged.append(currentStart..<endIndex)
            }
            return merged
        }

        switch behavior {
        case .removed:
            return split(by: string, options: options, includeSeparators: false)
        case .isolated:
            return split(by: string, options: options, includeSeparators: true)
        case .mergedWithNext:
            // Obtain ranges and merge them
            // "the-final--countdown" -> (3, 4), (9, 10), (10, 11) -> (start, 2), (3, 8), (9, 9), (10, end)
            let ranges = ranges(of: string, options: options)
            let merged = mergedWithNext(ranges: ranges)
            return merged.map { String(self[$0]) }
        case .mergedWithPrevious:
            // Obtain ranges and merge them
            // "the-final--countdown" -> (3, 4), (9, 10), (10, 11) -> (start, 3), (4, 9), (10, 10), (11, end)
            let ranges = ranges(of: string, options: options)
            let merged = mergedWithPrevious(ranges: ranges)
            return merged.map { String(self[$0]) }
        }
    }
}
