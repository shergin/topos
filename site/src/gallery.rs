//! The examples gallery: every example in the repository, with what
//! it printed into a pipe (recorded by `site/record.sh` into
//! `site/gallery/`) and its source, story first.

use std::fs;

use topos_plates::escape;
use topos_plates::highlight::highlight;

use crate::markdown::{self, Body, Context};

/// One example.
pub struct Entry {
    /// The `cargo run --example` name.
    pub name: &'static str,
    /// The source file, relative to `examples/`.
    pub path: &'static str,
    /// Extra arguments the recording passed.
    pub arguments: &'static str,
    /// The cargo features the recording was built with.
    pub features: &'static str,
    /// Whether `site/gallery/NAME.txt` exists; the rest show source only.
    pub recorded: bool,
    /// Why there is no recording, when there is none.
    pub note: &'static str,
}

/// One gallery section.
pub struct Section {
    pub id: &'static str,
    pub title: &'static str,
    pub intro: &'static str,
    pub entries: &'static [Entry],
}

const fn example(name: &'static str, path: &'static str) -> Entry {
    Entry {
        name,
        path,
        arguments: "",
        features: "",
        recorded: true,
        note: "",
    }
}

const fn unrecorded(name: &'static str, path: &'static str, note: &'static str) -> Entry {
    Entry {
        name,
        path,
        arguments: "",
        features: "",
        recorded: false,
        note,
    }
}

const fn model(
    name: &'static str,
    path: &'static str,
    arguments: &'static str,
    features: &'static str,
) -> Entry {
    Entry {
        name,
        path,
        arguments,
        features,
        recorded: true,
        note: "",
    }
}

/// The gallery, in reading order: from a scalar chain to a transformer.
pub const SECTIONS: &[Section] = &[
    Section {
        id: "the-stack-on-one-graph",
        title: "The stack on one graph",
        intro: "Where to start reading: the smallest graphs, and every reading of them printed in turn.",
        entries: &[
            example("walkthrough", "walkthrough.rs"),
            example("chain", "chain.rs"),
        ],
    },
    Section {
        id: "training-loops",
        title: "Training loops",
        intro: "One recording, many states. Each of these fits a small model with a loop that never touches the graph, and charts what it learned.",
        entries: &[
            example("gradient_descent", "gradient_descent.rs"),
            example("regression", "regression.rs"),
            example("mlp_xor", "mlp_xor.rs"),
            example("moons", "moons.rs"),
        ],
    },
    Section {
        id: "readings-and-seams",
        title: "Readings and seams",
        intro: "The research edge: a new AD mode and a new number type written outside the crate, graded bit for bit against the interpreter, and the backend ladder measured.",
        entries: &[
            example("forward_mode", "forward_mode.rs"),
            example("dual", "dual.rs"),
            example("element_seam", "element_seam.rs"),
            example("throughput", "throughput.rs"),
        ],
    },
    Section {
        id: "makemore",
        title: "makemore, act by act",
        intro: "Karpathy's classroom on a compiler: character-level language models on a list of names, from a bigram table to a small transformer, with the same act rewritten through facades, recorded gradients, Adam, data parallelism, batch norm, and emission.",
        entries: &[
            example("makemore_bigram", "makemore/bigram.rs"),
            example("makemore_mlp", "makemore/mlp.rs"),
            example("makemore_mlp_facade", "makemore/mlp_facade.rs"),
            example("makemore_mlp_compiled", "makemore/mlp_compiled.rs"),
            example("makemore_mlp_adam", "makemore/mlp_adam.rs"),
            example("makemore_mlp_parallel", "makemore/mlp_parallel.rs"),
            example("makemore_mlp_batchnorm", "makemore/mlp_batchnorm.rs"),
            example("makemore_embedding_map", "makemore/embedding_map.rs"),
            example("makemore_transformer", "makemore/transformer.rs"),
            unrecorded(
                "makemore_mlp_emitted",
                "makemore/mlp_emitted.rs",
                "This one trains through an emitted StableHLO step served by XLA, so it needs a Python with `jax` installed and is not recorded here.",
            ),
            unrecorded(
                "makemore_attention_grading",
                "makemore/attention_grading.rs",
                "A measurement harness over an XLA server; it needs a Python with `jax` installed and is not recorded here.",
            ),
        ],
    },
    Section {
        id: "vision",
        title: "Vision",
        intro: "Convolutional networks on real images: the im2col route, the plan tier under pressure, and the grading twins that settled the memory question.",
        entries: &[
            example("mnist", "mnist/main.rs"),
            example("cifar10", "cifar10/main.rs"),
            unrecorded(
                "mnist_grading",
                "mnist/grading.rs",
                "A measurement: one training route per process, timed and memory-monitored from outside. Its verdict is in the doc comment.",
            ),
            unrecorded(
                "cifar10_grading",
                "cifar10/grading.rs",
                "A measurement: one training route per process, timed and memory-monitored from outside. Its verdict is in the doc comment.",
            ),
        ],
    },
    Section {
        id: "language-models",
        title: "Language models",
        intro: "Released weights, recorded from the public op surface with no new opcodes. Each has its own page with the four engines, the measurements, and the cautionary tale.",
        entries: &[
            model(
                "gpt2",
                "gpt2/main.rs",
                "\"Once upon a time\" 40",
                "accelerate",
            ),
            model(
                "llama",
                "llama/main.rs",
                "\"Once upon a time\" 24",
                "accelerate",
            ),
        ],
    },
];

/// Every entry, flattened.
pub fn entries() -> impl Iterator<Item = &'static Entry> {
    SECTIONS.iter().flat_map(|section| section.entries.iter())
}

/// The entry with the given name.
pub fn entry(name: &str) -> Option<&'static Entry> {
    entries().find(|entry| entry.name == name)
}

/// The gallery name of a path under `examples/`, or a section id for a
/// directory that holds a series.
pub fn name_for(relative: &str) -> Option<String> {
    if let Some(entry) = entries().find(|entry| entry.path == relative) {
        return Some(entry.name.to_string());
    }
    match relative {
        "makemore" => Some("makemore".to_string()),
        "mnist" => Some("mnist".to_string()),
        "cifar10" => Some("cifar10".to_string()),
        _ => None,
    }
}

/// The example's source, with its doc comment separated from the code.
pub fn source(context: &Context, entry: &Entry) -> (String, String) {
    let path = context.site.root.join("examples").join(entry.path);
    let text =
        fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let mut doc = String::new();
    let mut code = String::new();
    let mut in_doc = true;
    for line in text.lines() {
        if in_doc {
            if let Some(rest) = line.strip_prefix("//!") {
                doc.push_str(rest.strip_prefix(' ').unwrap_or(rest));
                doc.push('\n');
                continue;
            }
            in_doc = false;
            if line.trim().is_empty() {
                continue;
            }
        }
        code.push_str(line);
        code.push('\n');
    }
    (doc.trim().to_string(), code.trim_end().to_string())
}

/// The recorded output of an example, if it was recorded.
fn recording(context: &Context, entry: &Entry) -> Option<String> {
    if !entry.recorded {
        return None;
    }
    let path = context
        .site
        .root
        .join("site/gallery")
        .join(format!("{}.txt", entry.name));
    let text = fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "{}: {error}\nrecord the gallery with ./site/record.sh",
            path.display()
        )
    });
    Some(text.trim_end_matches('\n').to_string())
}

/// The `cargo run` line that produced the recording.
fn command(entry: &Entry) -> String {
    let mut out = String::from("cargo run --release");
    if !entry.features.is_empty() {
        out.push_str(&format!(" --features {}", entry.features));
    }
    out.push_str(&format!(" --example {}", entry.name));
    if !entry.arguments.is_empty() {
        out.push_str(&format!(" -- {}", entry.arguments));
    }
    out
}

/// One entry as HTML: the story, the plate, and the source folded away.
pub fn entry_html(context: &Context, entry: &Entry, number: usize) -> String {
    let (doc, code) = source(context, entry);
    let story = markdown::fragment(&doc, context);
    let plate = match recording(context, entry) {
        Some(output) => {
            let tall = output.lines().count() > 40;
            format!(
                "<figure class=\"plate text{}\"><pre class=\"term\">{}</pre><figcaption><span class=\"plate-number\">Plate {number}.</span> What <code>{}</code> printed into a pipe.</figcaption></figure>\n",
                if tall { " tall" } else { "" },
                escape(&output),
                escape(&command(entry))
            )
        }
        None => format!(
            "<p class=\"unrecorded\">{}</p>\n",
            crate::figures::inline_markdown(entry.note)
        ),
    };
    format!(
        "<section class=\"example\" id=\"{name}\">\n<h3><a class=\"anchor-name\" href=\"#{name}\">{name}</a></h3>\n<div class=\"story\">{story}</div>\n{plate}\
         <details class=\"source\"><summary>The source, <code>examples/{path}</code></summary>\n<div class=\"code\" data-lang=\"Rust\"><pre><code>{code}</code></pre></div>\n<p class=\"source-link\"><a href=\"https://github.com/shergin/topos/blob/master/examples/{path}\">On GitHub</a></p></details>\n</section>\n",
        name = escape(entry.name),
        path = escape(entry.path),
        code = highlight("rust", &code),
    )
}

/// The gallery page.
pub fn render(context: &Context) -> Body {
    let mut html = String::new();
    let mut headings = Vec::new();
    let mut text = String::new();
    html.push_str("<p>Every example in the repository, in reading order. Each plate is the exact text <code>cargo run --release --example NAME</code> printed into a pipe: no color, a fixed width, the charts as the plain grid a log file or a language model reads. The story above each plate is the file's own doc comment, and the source is folded beneath it.</p>\n");
    html.push_str("<p>The recordings live in <code>site/gallery/</code> and are remade by <code>./site/record.sh</code>. Two of the language-model runs need a cached checkpoint and the <code>accelerate</code> build; the rest run from a clean checkout.</p>\n");
    html.push_str("<nav class=\"gallery-index\" aria-label=\"Sections\"><ol>\n");
    for section in SECTIONS {
        html.push_str(&format!(
            "<li><a href=\"#{}\">{}</a> <span>{}</span></li>\n",
            section.id,
            escape(section.title),
            section.entries.len()
        ));
    }
    html.push_str("</ol></nav>\n");
    let mut number = 0;
    for section in SECTIONS {
        html.push_str(&format!(
            "<h2 id=\"{id}\">{}<a class=\"anchor\" href=\"#{id}\" aria-label=\"Link to this section\">#</a></h2>\n",
            escape(section.title),
            id = section.id
        ));
        html.push_str(&format!(
            "<p class=\"section-intro\">{}</p>\n",
            crate::figures::inline_markdown(section.intro)
        ));
        headings.push(markdown::Heading {
            level: 2,
            id: section.id.to_string(),
            text: section.title.to_string(),
        });
        for entry in section.entries {
            number += 1;
            html.push_str(&entry_html(context, entry, number));
            let (doc, _) = source(context, entry);
            text.push_str(entry.name);
            text.push(' ');
            text.push_str(&doc);
            text.push(' ');
        }
    }
    Body {
        html,
        headings,
        summary: text.chars().take(1500).collect(),
        scripts: Vec::new(),
    }
}
