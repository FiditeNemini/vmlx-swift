# Vendored XGrammar bridge

Swift bridge donor: ml-explore/mlx-swift-lm commit22157fc397b59acfb03e91c370bcbf2cfb10970e (MLXCXGrammar target only).
C++ source: XGrammar v0.1.30, upstream commit d476a48dcd8fa3b5afeddbe850e73bb3b1dcf505, retained under xgrammar/. LICENSE and NOTICE remain in that directory; picojson/DLPack retain embedded notices.

No guided-generation loop, completion/whitespace biases, tokenizer heuristics or prompt changes are imported. The local shim JSON Schema compiler explicitly sets strict_mode=false so omitted additionalProperties/items follow JSON Schema defaults. The Swift admission layer rejects unsupported schema constraints before compilation. Vendored C++ source is otherwise unchanged. The grammar_functor wrapper retains the donor static-member link fix.

The namespace macros isolate C++ xgrammar and picojson symbols. The existing package C++20 setting compiles the donor's C++17-compatible source; no global language-standard change is introduced.

Local C++20 compatibility: moved the SequenceFormat and OrFormat constructor bodies after all recursive variant alternatives in structural_tag.h. No grammar behavior change.
