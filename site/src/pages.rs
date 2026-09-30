//! The page table: every URL on the site, what it is built from, and
//! how the navigation groups it.

/// Where a page's body comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// A markdown file under `site/content/`.
    Content(&'static str),
    /// A markdown file in the repository, rendered with its links rewritten.
    Repo(&'static str),
    /// A repository markdown file with site material inserted after
    /// named headings: the repository stays the single source of the
    /// text, the site adds the plates.
    RepoWith(&'static str, &'static [(&'static str, &'static str)]),
    /// The front page.
    Home,
    /// The examples gallery, built from the recorded outputs and the
    /// example sources.
    Gallery,
    /// The playground.
    Playground,
}

/// One page.
#[derive(Debug, Clone, Copy)]
pub struct Page {
    /// The site-absolute URL, always a directory (`/guide/plans/`).
    pub url: &'static str,
    /// The title, as the navigation and the browser tab show it.
    pub title: &'static str,
    /// One line for the section index and the search results.
    pub blurb: &'static str,
    /// Where the body comes from.
    pub source: Source,
}

impl Page {
    /// The output path relative to the output directory.
    pub fn path(&self) -> String {
        format!("{}index.html", self.url.trim_start_matches('/'))
    }

    /// The relative prefix that reaches the site root from this page.
    pub fn root(&self) -> String {
        let depth = self
            .url
            .trim_matches('/')
            .split('/')
            .filter(|part| !part.is_empty())
            .count();
        "../".repeat(depth)
    }

    /// The repository directory relative links in this page's source
    /// resolve against.
    pub fn source_dir(&self) -> &'static str {
        match self.source {
            Source::Repo(path) | Source::RepoWith(path, _) => {
                path.rsplit_once('/').map_or("", |(dir, _)| dir)
            }
            _ => "",
        }
    }

    /// Whether the search index lists the page.
    pub fn searchable(&self) -> bool {
        !matches!(self.source, Source::Home)
    }
}

/// A navigation section.
pub struct Section {
    pub title: &'static str,
    pub pages: &'static [Page],
}

const fn page(url: &'static str, title: &'static str, blurb: &'static str, source: Source) -> Page {
    Page {
        url,
        title,
        blurb,
        source,
    }
}

/// The navigation, in order.
pub const SECTIONS: &[Section] = &[
    Section {
        title: "Start here",
        pages: &[
            page(
                "/",
                "topos",
                "An autodiff compiler stack in Rust. Record a graph, inspect it, differentiate it, compile it, emit it.",
                Source::Home,
            ),
            page(
                "/guide/start/",
                "Getting started",
                "Install, the first tape, a gradient, a training step, and the readings of one small graph.",
                Source::Content("guide/start.md"),
            ),
            page(
                "/walkthrough/",
                "The stack, read six ways",
                "One graph — a matrix product, a tanh, a loss — as the spec, its derivative, a plan, StableHLO, two runs that must agree, and a walk by hand.",
                Source::Content("walkthrough.md"),
            ),
            page(
                "/playground/",
                "Playground",
                "Write a small graph in the browser and read it six ways, live: the spec, the gradient, the plan, the StableHLO, the values, the picture.",
                Source::Playground,
            ),
        ],
    },
    Section {
        title: "The guide",
        pages: &[
            page(
                "/guide/recording/",
                "Recording",
                "Tapes, values, symbols, and shapes checked at the line that records them. Sealing, reopening, and the IR dump.",
                Source::Content("guide/recording.md"),
            ),
            page(
                "/guide/differentiation/",
                "Differentiation",
                "The engine reverse scan, the same rules recorded as more spec, seeds and VJPs, second derivatives, and forward mode as an outside reading.",
                Source::Content("guide/differentiation.md"),
            ),
            page(
                "/guide/training/",
                "Training",
                "The spec is immutable and the state is yours: parameters, steps, optimizers, what-ifs by cloning, checkpoints, and the curves.",
                Source::Content("guide/training.md"),
            ),
            page(
                "/guide/plans/",
                "Entries and plans",
                "Declare what a run may read. A plan is the schedule that declaration licenses: liveness, skipped nodes, elected patterns, exact and fast.",
                Source::Content("guide/plans.md"),
            ),
            page(
                "/guide/emission/",
                "Emission",
                "The plan as StableHLO text: what lowers, how a raise turns unfold-and-multiply into a convolution, and how the oracle checks the other side.",
                Source::Content("guide/emission.md"),
            ),
            page(
                "/guide/facades/",
                "The neural tier",
                "Layers, losses, attention, dropout, optimizers: every facade is a spelling over the public operations, and the dump proves it.",
                Source::Content("guide/facades.md"),
            ),
            page(
                "/guide/elements/",
                "The element seam",
                "f32, f64, Bf16, and the number you bring: one graph kind, and a new number inherits the whole stack.",
                Source::Content("guide/elements.md"),
            ),
            page(
                "/guide/acceleration/",
                "Acceleration",
                "Opt-in backends, the measured ladder, Fast against Exact, and the tally that reports who served.",
                Source::RepoWith(
                    "docs/acceleration.md",
                    &[
                        (
                            "## Turn it on",
                            "{{figure accel_status | What the backends answer in the build that made this page: no feature is on, so every one declines and the interpreter serves. Turn a feature on and the same line reports `ready`.}}",
                        ),
                        (
                            "## Measured",
                            "{{figure accel_products nocode | The products row of the table above, drawn from its numbers: one bar per build, on a log axis, because the ladder spans two orders of magnitude.}}",
                        ),
                        (
                            "## Fast and exact",
                            "{{figure plan_exact_fast | Three losses from one graph: the interpreter, an `Exact` plan, and the default `Fast` plan. In a build with no backend all three agree bit for bit; the assertion between the first two holds in every build.}}",
                        ),
                    ],
                ),
            ),
            page(
                "/guide/notebooks/",
                "Notebooks",
                "Evcxr, the two rules a cell must follow, and the card every type draws itself as.",
                Source::RepoWith(
                    "docs/notebooks.md",
                    &[(
                        "## Cell output",
                        "{{figure card_network nocode | A `Network` at the end of a cell: the spec dump, exactly as `describe` prints it, in the card the notebook shows. Every card on this page is the HTML `to_html` emitted, embedded as is.}}\n\n{{figure card_plan nocode | A `Plan`: the schedule, then the live volume along it.}}\n\n{{figure card_tensor nocode | A `Tensor` small enough to print exactly, shaded by value.}}\n\n{{figure card_field nocode | A `Field` of gradients: one Euclidean norm per node, along the tape. A vanishing or exploding region is a shape, not a number to hunt for.}}\n\n{{figure card_parameters nocode | `Parameters`: each slot's shape, and the value when it is a scalar.}}\n\n{{figure card_adjoints nocode | `Adjoints`: the target and each `wrt → gradient` pair.}}\n\n{{figure card_entry nocode | An `Entry`: roots, observes, memory posture, numerics.}}",
                    )],
                ),
            ),
        ],
    },
    Section {
        title: "Examples",
        pages: &[
            page(
                "/examples/",
                "The gallery",
                "Every example in the repository, with what it printed and its source: from a scalar chain to a transformer.",
                Source::Gallery,
            ),
            page(
                "/examples/gpt2/",
                "GPT-2 on topos",
                "The released 124M weights, recorded from the public op surface, generated four ways — and a vendor GPU plugin caught being wrong.",
                Source::Repo("examples/gpt2/README.md"),
            ),
            page(
                "/examples/llama/",
                "Llama on topos",
                "TinyLlama and Llama 2 7B from one module tree: RMS norm, rotary embeddings, grouped queries, SwiGLU, all as composition.",
                Source::Repo("examples/llama/README.md"),
            ),
        ],
    },
    Section {
        title: "Why it is shaped this way",
        pages: &[
            page(
                "/principles/",
                "Vision",
                "The argument, the three commitments, and the five rules.",
                Source::RepoWith(
                    "docs/vision.md",
                    &[(
                        "# Vision",
                        "{{figure hero nocode | One small graph and its derivative, drawn from the spec. The lighter nodes are the ones `differentiate` appended: the chain rule as more of the same graph.}}",
                    )],
                ),
            ),
            page(
                "/principles/recorded-reverse/",
                "Recorded reverse mode",
                "Reverse mode is a transform of the spec, not an engine procedure that forgets the math.",
                Source::RepoWith(
                    "docs/principles/recorded-reverse.md",
                    &[(
                        "## The idea",
                        "{{figure diff_equal nocode | The two readings of one rule body, asserted equal: the engine scan computes a gradient, `differentiate` records the same rules as nodes, and a run of those nodes answers the same bits.}}",
                    )],
                ),
            ),
            page(
                "/principles/spec-and-state/",
                "Spec and state",
                "The architecture is an immutable value. The weights are yours. Training does not version the graph.",
                Source::RepoWith(
                    "docs/principles/spec-and-state.md",
                    &[(
                        "## The idea",
                        "{{figure train_state nocode | A hundred steps later the network still prints the same six lines, and `network.parameters()` still answers the initials. Only the table the loop holds moved.}}",
                    )],
                ),
            ),
            page(
                "/principles/names/",
                "Names",
                "A name, a structural witness, and the right to extend the recording are three jobs, not one identity.",
                Source::RepoWith(
                    "docs/principles/names.md",
                    &[(
                        "## The idea",
                        "{{figure rec_symbols nocode | A `Value` records; a `Symbol` is the detached name every later phase reads through. The proxy dies at the seal, the name does not.}}",
                    )],
                ),
            ),
            page(
                "/principles/vocabulary/",
                "What earns an instruction",
                "An instruction sits in the vocabulary iff something real speaks it and no composition of the rest reproduces its bits.",
                Source::RepoWith(
                    "docs/principles/vocabulary.md",
                    &[(
                        "## The idea",
                        "{{figure facade_attention nocode | Attention did not earn an instruction. One head is a handful of the existing ones — the stable softmax is the only fused core in the spelling.}}",
                    )],
                ),
            ),
            page(
                "/principles/element/",
                "The element is the seam",
                "The graph is always tensors. A scalar is rank 0. The open plug is a number, not a second engine.",
                Source::RepoWith(
                    "docs/principles/element.md",
                    &[(
                        "## The idea",
                        "{{figure elem_precision nocode | One recording, three numbers. The spec is the same text over every element; only the bits a run answers differ, and each element's bits are its own.}}",
                    )],
                ),
            ),
            page(
                "/principles/observability/",
                "Observability is a license",
                "What you may read is declared. Everything derived runs on that declaration, not on the whole tape.",
                Source::RepoWith(
                    "docs/principles/observability.md",
                    &[(
                        "## The idea",
                        "{{figure plan_observe nocode | The same graph lowered twice. Declaring `hidden` readable turns its line from `freed` to `kept`; nothing else about the schedule changes.}}",
                    )],
                ),
            ),
            page(
                "/openings/",
                "Openings",
                "Decisions and what they paid for: not rules, consequences.",
                Source::Repo("docs/openings/README.md"),
            ),
            page(
                "/openings/ad-as-a-reading/",
                "AD as a named reading",
                "Two usual walks over one graph, and a third one written outside the crate.",
                Source::RepoWith(
                    "docs/openings/ad-as-a-reading.md",
                    &[(
                        "## What it opened",
                        "{{figure diff_hessian nocode | A second derivative is reverse mode of a gradient. The gradient was recorded as ordinary nodes, so `vjp` walks it like anything else, and the Hessian-vector product is one more root.}}",
                    )],
                ),
            ),
            page(
                "/openings/recurrence-as-feeds/",
                "Recurrence as feeds",
                "A decode cache has to live somewhere. It lives where a batch lives.",
                Source::Repo("docs/openings/recurrence-as-feeds.md"),
            ),
            page(
                "/openings/several-exports/",
                "Several exports, one spec",
                "Train and sample share weights. They should not compute the same nodes.",
                Source::RepoWith(
                    "docs/openings/several-exports.md",
                    &[(
                        "## What it opened",
                        "{{figure open_exports nocode | Two entries over one sealed network: the training entry never schedules the sampling head, and the sampling entry never schedules the loss.}}",
                    )],
                ),
            ),
            page(
                "/openings/fusion-raises/",
                "Fusion raises",
                "The tape stays the spec. A faster form is an offer, and the same match hands another compiler its named op.",
                Source::RepoWith(
                    "docs/openings/fusion-raises.md",
                    &[(
                        "## What it opened",
                        "{{figure emit_conv nocode | A convolution recorded as pad, unfold, permute, reshape, and a matrix product — and emitted as `stablehlo.convolution`, because the matcher found the idiom again.}}",
                    )],
                ),
            ),
            page(
                "/openings/the-oracle-ships/",
                "The oracle ships",
                "The slow, obvious run is the spec, and it checks the other side of the boundary once the plan leaves the crate.",
                Source::RepoWith(
                    "docs/openings/the-oracle-ships.md",
                    &[(
                        "## What it opened",
                        "{{figure plan_exact_fast nocode | The interpreter, an exact plan, and a fast plan on one graph. The first two are asserted equal in every build.}}",
                    )],
                ),
            ),
            page(
                "/openings/dropout-as-a-feed/",
                "Dropout as a feed",
                "No generator in the graph and no train/eval switch. Inference is the absence of a feed.",
                Source::RepoWith(
                    "docs/openings/dropout-as-a-feed.md",
                    &[(
                        "## What it opened",
                        "{{figure facade_dropout nocode | The same `Dropout` expression run twice: unfed, the all-ones default makes it the identity; fed a seeded mask, it drops and rescales.}}",
                    )],
                ),
            ),
        ],
    },
    Section {
        title: "Reference",
        pages: &[
            page(
                "/concepts/",
                "Terminology",
                "The public vocabulary: each entry's literature meaning and where it lives here, illustrated.",
                Source::RepoWith(
                    "docs/terminology.md",
                    &[
                        ("## Recording", "{{figure rec_sources nocode}}"),
                        ("## Running", "{{figure plan_observe nocode}}"),
                        ("## Differentiation", "{{figure diff_recorded nocode}}"),
                        ("## Payload", "{{figure elem_precision nocode}}"),
                        ("## Acceleration", "{{figure accel_status nocode}}"),
                        ("## Facades", "{{figure facade_mlp nocode}}"),
                    ],
                ),
            ),
            page(
                "/changelog/",
                "Changelog",
                "Every release, written for a person.",
                Source::Repo("CHANGELOG.md"),
            ),
        ],
    },
];

/// Every page, flattened.
pub fn all() -> impl Iterator<Item = &'static Page> {
    SECTIONS.iter().flat_map(|section| section.pages.iter())
}

/// The site URL a repository path is published at, if any.
pub fn url_for(repo_path: &str) -> Option<String> {
    let path = repo_path.trim_start_matches("./").trim_end_matches('/');
    let fixed = match path {
        "README.md" | "site/README.md" => Some("/"),
        "docs/vision.md" => Some("/principles/"),
        "docs/terminology.md" => Some("/concepts/"),
        "docs/acceleration.md" => Some("/guide/acceleration/"),
        "docs/notebooks.md" => Some("/guide/notebooks/"),
        "docs/principles" => Some("/principles/"),
        "docs/openings" | "docs/openings/README.md" => Some("/openings/"),
        "examples" => Some("/examples/"),
        "examples/walkthrough.rs" => Some("/walkthrough/"),
        "examples/gpt2" | "examples/gpt2/README.md" => Some("/examples/gpt2/"),
        "examples/llama" | "examples/llama/README.md" => Some("/examples/llama/"),
        "CHANGELOG.md" => Some("/changelog/"),
        _ => None,
    };
    if let Some(url) = fixed {
        return Some(url.to_string());
    }
    if let Some(page) = all().find(
        |page| matches!(page.source, Source::Repo(source) | Source::RepoWith(source, _) if source == path),
    ) {
        return Some(page.url.to_string());
    }
    // An example's source links to its gallery entry.
    if let Some(rest) = path.strip_prefix("examples/") {
        let name = crate::gallery::name_for(rest)?;
        return Some(format!("/examples/#{name}"));
    }
    None
}
