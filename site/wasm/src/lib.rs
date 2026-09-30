//! The playground's engine: a small tape language, recorded onto a
//! real `Tape` and read six ways — the spec, the derivative, the plan,
//! the StableHLO, the values, the picture — with every reading
//! rendered by the same plates the site uses.
//!
//! The language is a thin spelling of what `Tape` and `Value` already
//! offer; nothing here computes anything the crate does not. Shapes
//! are checked before a line records, so a mistake is a message in
//! the page rather than a panic in the wasm.

mod language;

use serde::Serialize;
use topos::{Numerics, Tape};
use topos_plates::graph::Graph;
use topos_plates::highlight::highlight;
use topos_plates::ir::{self, Dump};
use topos_plates::{escape, graph};
use wasm_bindgen::prelude::*;

/// One named node, for the page's controls.
#[derive(Serialize)]
pub struct Named {
    pub name: String,
    pub index: usize,
    pub shape: String,
    pub scalar: bool,
    pub parameter: bool,
}

/// Every reading of one program, as HTML the page inserts.
#[derive(Serialize, Default)]
pub struct Readings {
    pub names: Vec<Named>,
    pub spec: String,
    pub derivative: String,
    pub plan: String,
    pub hlo: String,
    pub values: String,
    pub graph: String,
    pub summary: String,
    pub error: Option<String>,
}

/// Records `program` and answers its readings as JSON. `target` names
/// the rank-0 node to differentiate and to root the plan on (empty
/// for the last statement), `observe` an extra name to declare
/// readable (empty for none), and `exact` the numerics posture.
#[wasm_bindgen]
pub fn readings(program: &str, target: &str, observe: &str, exact: bool) -> String {
    let readings = read(program, target, observe, exact);
    serde_json::to_string(&readings)
        .unwrap_or_else(|error| format!("{{\"error\":\"{}\"}}", escape(&error.to_string())))
}

fn read(program: &str, target: &str, observe: &str, exact: bool) -> Readings {
    let tape: Tape<f32> = Tape::new();
    let recorded = match language::record(&tape, program) {
        Ok(recorded) => recorded,
        Err(error) => {
            return Readings {
                error: Some(error),
                ..Readings::default()
            };
        }
    };
    let names: Vec<Named> = recorded
        .names
        .iter()
        .map(|(name, symbol, shape)| Named {
            name: name.clone(),
            index: symbol.index(),
            shape: shape.to_string(),
            scalar: shape.rank() == 0,
            parameter: recorded.parameters.contains(symbol),
        })
        .collect();
    if names.is_empty() {
        return Readings {
            error: Some("write at least one line, such as `w = parameter 0.5`".to_string()),
            ..Readings::default()
        };
    }
    let spec_text = tape.describe();
    let spec_nodes = tape.len();

    // The target: the named rank-0 node, else the last rank-0 name.
    let chosen = names
        .iter()
        .find(|named| named.name == target)
        .or_else(|| names.iter().rev().find(|named| named.scalar))
        .or_else(|| names.last())
        .expect("at least one name");
    let target_symbol = recorded
        .symbol(&chosen.name)
        .expect("a listed name resolves");
    let observed = names
        .iter()
        .find(|named| named.name == observe && named.name != chosen.name)
        .and_then(|named| recorded.symbol(&named.name));

    // The derivative: only a rank-0 target has a gradient, and only
    // parameters are differentiated with respect to.
    let adjoints = if chosen.scalar && !recorded.parameters.is_empty() {
        Some(tape.differentiate(target_symbol, recorded.parameters.clone()))
    } else {
        None
    };
    let derivative = match &adjoints {
        Some(adjoints) => {
            let appended = tape.len() - spec_nodes;
            let pairs: Vec<String> = adjoints
                .pairs()
                .iter()
                .map(|(wrt, gradient)| {
                    format!(
                        "d {} / d {} is node {}",
                        chosen.name,
                        recorded.name_of(*wrt),
                        gradient.index()
                    )
                })
                .collect();
            let label = format!(
                "{appended} nodes appended by differentiate: {}",
                pairs.join(", ")
            );
            format!(
                "<pre class=\"term ir\">{}</pre>",
                ir::to_html(&Dump {
                    text: &tape.describe(),
                    split: Some((spec_nodes, &label)),
                    marks: &[],
                })
            )
        }
        None if !chosen.scalar => note(&format!(
            "`{}` has shape {} and a gradient needs a rank-0 target: wrap it in `sum(...)`, or pick another target.",
            chosen.name, chosen.shape
        )),
        None => note(
            "there is no parameter to differentiate with respect to: declare one with `w = parameter [2, 2]`.",
        ),
    };

    let network = tape.into_network();
    let parameters = network.parameters();

    // The plan over the declared reading.
    let roots: Vec<topos::Symbol> = match &adjoints {
        Some(adjoints) => adjoints.roots().collect(),
        None => vec![target_symbol],
    };
    let mut entry = network.entry(roots.clone());
    if let Some(observed) = observed {
        entry = entry.observe([observed]);
    }
    if exact {
        entry = entry.numerics(Numerics::Exact);
    }
    let plan = entry.lower();
    let results: Vec<usize> = plan.results().iter().map(|symbol| symbol.index()).collect();
    let plan_html = format!(
        "<pre class=\"term ir\">{}</pre>",
        ir::to_html(&Dump {
            text: &plan.describe(),
            split: None,
            marks: &results,
        })
    );
    let hlo = match plan.emit_stablehlo() {
        Ok(module) => format!(
            "<div class=\"code hlo\" data-lang=\"StableHLO\"><pre><code>{}</code></pre></div>",
            highlight("mlir", module.trim_end())
        ),
        Err(error) => note(&format!("this plan does not emit: {error}")),
    };

    // The values: the interpreter over the whole spec, read by name.
    let run = network.forward(&parameters, []);
    let mut lines: Vec<String> = names
        .iter()
        .map(|named| {
            let symbol = recorded.symbol(&named.name).expect("listed");
            format!("{:<12} {:<10} {}", named.name, named.shape, run.of(symbol))
        })
        .collect();
    if let Some(adjoints) = &adjoints {
        lines.push(String::new());
        for (wrt, gradient) in adjoints.pairs() {
            lines.push(format!(
                "d {} / d {:<6} {}",
                chosen.name,
                recorded.name_of(*wrt),
                run.of(*gradient)
            ));
        }
    }
    let values = format!("<pre class=\"term\">{}</pre>", escape(&lines.join("\n")));

    // The picture: the spec, its derivative shaded, the unscheduled faded.
    let scheduled: Vec<usize> = plan.nodes().map(|node| node.symbol().index()).collect();
    let mut picture = Graph::from_nodes(network.nodes()).classify_from(spec_nodes, "derived");
    picture = picture.classify_others(&scheduled, "skipped");
    if let Some(observed) = observed {
        picture = picture.classify(&[observed.index()], "kept");
    }
    let graph_html = format!(
        "<div class=\"graph-plate\">{}</div>",
        graph::svg(&picture, "the recorded graph")
    );

    let summary = format!(
        "{} nodes recorded, {} scheduled, {} readable; differentiating {}{}",
        network.len(),
        plan.len(),
        results.len(),
        chosen.name,
        match observed {
            Some(observed) => format!(", observing {}", recorded.name_of(observed)),
            None => String::new(),
        }
    );
    Readings {
        names,
        spec: format!(
            "<pre class=\"term ir\">{}</pre>",
            ir::to_html(&Dump::new(&spec_text))
        ),
        derivative,
        plan: plan_html,
        hlo,
        values,
        graph: graph_html,
        summary,
        error: None,
    }
}

fn note(text: &str) -> String {
    format!("<p class=\"status\">{}</p>", escape(text))
}
