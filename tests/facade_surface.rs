//! The facade tier gate: everything under `src/facade/` composes
//! through the public surface alone.
//!
//! The tier doctrine ([terminology](../docs/terminology.md): Facades)
//! promises that every facade is an ordinary consumer — no privileged
//! engine access — so a
//! hand-rolled equivalent behaves identically and internals may change
//! their spelling freely underneath. The compiler cannot police this
//! inside one crate, so this test does: it scans the facade sources for
//! the two ways privilege has actually leaked — naming a private module
//! path, and suppressing the lints that fire when a crate-private item
//! hides inside a public signature.
//!
//! It is a lint over source text, not a proof: a facade could still
//! call a `pub(crate)` method without naming any module path. The house
//! import rules (imports at the top of the file, no inline qualified
//! paths) are what make the scan sound in practice, and the suppression
//! check catches the historical counterexample (the notebook tier's
//! `private_bounds` allows). The method-call shape has its own
//! historical example — the notebook once read `Entry`'s fields,
//! `ValueId::index`, `Parameters::payloads`, and `Tensor::as_constant`
//! before they were published as designed reads — and its own gate:
//! `notebook_surface.rs` compiles as an external consumer and rebuilds
//! each card's data from public readers alone. The scan reads the
//! files from disk, so the `notebook` tier is covered even when the
//! `evcxr` feature that compiles it is off.

use std::fs;
use std::path::{Path, PathBuf};

/// The directories holding the facade tier.
const FACADE_DIRECTORIES: [&str; 2] = ["src/facade/neural", "src/facade/notebook"];

/// The crate's private modules: naming one from a facade is privileged
/// access, whatever the item behind it. Both spellings count -- the
/// flat re-export (`crate::graph`) and the tier folder it lives in
/// (`crate::core`) -- because a private module at the crate root is
/// reachable from every descendant. The public roads are the crate
/// root's re-exports and the `reference`, `model`, and `compiler`
/// modules.
const INTERNAL_MODULES: [&str; 11] = [
    "backend", "core", "derived", "emission", "engine", "facade", "graph", "neural", "notebook",
    "op", "payload",
];

/// The lint suppressions that would let a crate-private item hide
/// inside a facade's public signatures.
const LEAK_SUPPRESSIONS: [&str; 2] = ["private_bounds", "private_interfaces"];

/// Collects every Rust source under `directory`, tests included: the
/// tier's tests are held to the same bar as the tier itself.
fn rust_sources(directory: &Path, sources: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(directory).expect("facade directory is readable");
    for entry in entries {
        let path = entry.expect("directory entry is readable").path();
        if path.is_dir() {
            rust_sources(&path, sources);
            continue;
        }
        if path.extension().is_some_and(|extension| extension == "rs") {
            sources.push(path);
        }
    }
}

/// Returns the identifier starting at `text`, read as far as identifier
/// characters continue.
fn leading_identifier(text: &str) -> &str {
    let end = text
        .find(|character: char| !character.is_alphanumeric() && character != '_')
        .unwrap_or(text.len());
    &text[..end]
}

/// Scans one file's text for `crate::<internal module>` paths and the
/// leak suppressions, answering one `file:line` record per finding.
fn violations_in(path: &Path, text: &str) -> Vec<String> {
    let mut violations = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let mut rest = line;
        while let Some(position) = rest.find("crate::") {
            let after = &rest[position + "crate::".len()..];
            let segment = leading_identifier(after);
            if INTERNAL_MODULES.contains(&segment) {
                violations.push(format!(
                    "{}:{}: names the private module `crate::{segment}`",
                    path.display(),
                    index + 1
                ));
            }
            rest = after;
        }
        for suppression in LEAK_SUPPRESSIONS {
            if line.contains(suppression) {
                violations.push(format!(
                    "{}:{}: suppresses `{suppression}`",
                    path.display(),
                    index + 1
                ));
            }
        }
    }
    violations
}

#[test]
fn the_gate_recognizes_both_leak_shapes() {
    let text = "use crate::graph::network::Network;\n#[allow(private_bounds)]\nuse crate::Tape;";
    let found = violations_in(Path::new("synthetic.rs"), text);
    assert_eq!(found.len(), 2, "one module path plus one suppression");
}

#[test]
fn facades_compose_through_the_public_surface_alone() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut sources = Vec::new();
    for directory in FACADE_DIRECTORIES {
        rust_sources(&root.join(directory), &mut sources);
    }
    assert!(
        !sources.is_empty(),
        "the facade directories moved; update this gate"
    );

    let mut violations = Vec::new();
    for path in &sources {
        let text = fs::read_to_string(path).expect("facade source is readable");
        violations.extend(violations_in(path, &text));
    }
    assert!(
        violations.is_empty(),
        "facades must compose through the public surface alone; \
         publish the read the facade needs instead of reaching through:\n{}",
        violations.join("\n")
    );
}
