// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Template-context adapter for Muse Glimmer (`model_type = muse_glimmer`).
///
/// Muse Glimmer keys reasoning on **`reasoning_strength`**, not the
/// `reasoning_effort` every other family understands. Its `render_reasoning()`
/// writes `Reasoning strength: <value>.` into the SYSTEM PREFIX and defaults to
/// `high` when the variable is undefined:
///
/// ```jinja
/// {%- set rs = reasoning_strength if reasoning_strength is defined and reasoning_strength else 'high' -%}
/// {{- 'Reasoning strength: ' + rs + '.' -}}
/// ```
///
/// Nothing in the library ever set that key. `reasoning_strength` appeared only
/// in this repo's own docs and in `MediaSalt`, which merely READS it out of
/// `additionalContext` for the cache scope salt — and, in osaurus, only in test
/// files. So the shipping app sent `reasoning_effort`, the template ignored it,
/// and **every Muse Glimmer request ran at `high`** regardless of what the user
/// chose. The control was editable, saved, and inert.
///
/// `MuseGlimmer_PROOF_MATRIX.md:250` already predicted this in writing — "If
/// osaurus sends `reasoning_effort`, the template never…" — which is why the
/// translation belongs in code rather than in another comment.
///
/// **Four levels, not three.** The official contract is
/// `low / medium / high / xhigh`, with `high`/`xhigh` intended for complex
/// problem solving, coding and agentic work. A caller that models effort as
/// low/medium/high only cannot express `xhigh`, so `max` maps there rather than
/// being flattened into `high` and silently capping the model below its ceiling.
///
/// **There is no "off".** Reasoning is a system-prefix sentence, not a channel:
/// there are no `<think>` tags to suppress. The lowest expressible setting is
/// `low`, so a request to disable thinking maps to `low` rather than removing
/// the key — removing it would fall through to the template's `high` default,
/// i.e. the exact opposite of what was asked.
public enum MuseGlimmerReasoningTemplateContext {

    /// Accepted values, in the order the model card documents them.
    public static let levels = ["low", "medium", "high", "xhigh"]

    public static func applies(to modelType: String?) -> Bool {
        guard let t = modelType?.lowercased() else { return false }
        return t == "muse_glimmer" || t.hasPrefix("muse_glimmer") || t.hasPrefix("museglimmer")
    }

    public static func apply(
        additionalContext: [String: any Sendable]?,
        modelType: String?
    ) -> [String: any Sendable]? {
        guard applies(to: modelType) else { return additionalContext }
        var context = additionalContext ?? [:]

        // An explicit `reasoning_strength` from the caller is already in the
        // template's own vocabulary — normalise it but never re-derive it from
        // `reasoning_effort`, which would let a generic default override a
        // deliberate per-model choice.
        if let explicit = (context["reasoning_strength"] as? String)?.lowercased(),
            !explicit.isEmpty
        {
            context["reasoning_strength"] = normalized(explicit)
            return context
        }

        if let requested = (context["reasoning_effort"] as? String)?.lowercased(),
            !requested.isEmpty
        {
            context["reasoning_strength"] = normalized(requested)
        } else if let enable = context["enable_thinking"] as? Bool {
            // No channel to switch off, so "off" is the floor, not absence.
            context["reasoning_strength"] = enable ? "high" : "low"
        }
        // With neither present, leave the key unset and let the template's own
        // `high` default apply — the documented behaviour for an undeclared
        // request.
        return context.isEmpty ? nil : context
    }

    /// Map the request surface onto the four documented levels.
    static func normalized(_ raw: String) -> String {
        switch raw {
        case "low", "medium", "high", "xhigh": return raw
        // OpenAI-style and internal synonyms.
        case "minimal", "none", "off", "false", "no_think", "instruct": return "low"
        case "max", "highest", "ultra": return "xhigh"
        default: return "high"
        }
    }
}
