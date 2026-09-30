//! What a figure block hands the site to render: one of the crate's
//! readings, named by the way it should be shown.
//!
//! A figure file ends with one of these constructors —
//! `show::spec(network.describe())`, `show::plot(chart)`,
//! `show::graph(Graph::from_nodes(network.nodes()))` — so the block
//! reads as ordinary topos code with a final "and show this".

use malevich::Plot;
use topos_plates::graph::Graph;

/// A reading, and the form it takes on the page.
pub enum Exhibit {
    /// A spec or schedule dump: the text `describe` printed, shown as
    /// an IR plate with every operand reference linked.
    Spec {
        text: String,
        /// The first index of an appended region and its divider label.
        split: Option<(usize, String)>,
        /// Node indices to underline.
        marks: Vec<usize>,
    },
    /// A StableHLO module, highlighted.
    Hlo(String),
    /// Plain program output, as a terminal plate.
    Text(String),
    /// A chart, as the SVG terminal card malevich draws.
    Plot(Box<Plot<'static>>),
    /// A graph picture.
    Graph(Graph),
    /// A notebook card: the HTML `to_html` emitted, embedded as is.
    Card(String),
    /// Several readings under tabs.
    Panels(Vec<(String, Exhibit)>),
}

/// A spec or plan dump.
pub fn spec(text: impl Into<String>) -> Exhibit {
    Exhibit::Spec {
        text: text.into(),
        split: None,
        marks: Vec::new(),
    }
}

/// A dump whose lines from `split` on were appended by a transform or
/// a reopen; `label` is drawn as the divider above them.
pub fn spec_from(text: impl Into<String>, split: usize, label: &str) -> Exhibit {
    Exhibit::Spec {
        text: text.into(),
        split: Some((split, label.to_string())),
        marks: Vec::new(),
    }
}

/// A dump with the given node indices underlined.
pub fn spec_marked(text: impl Into<String>, marks: impl IntoIterator<Item = usize>) -> Exhibit {
    Exhibit::Spec {
        text: text.into(),
        split: None,
        marks: marks.into_iter().collect(),
    }
}

/// A StableHLO module.
pub fn hlo(text: impl Into<String>) -> Exhibit {
    Exhibit::Hlo(text.into())
}

/// Plain program output.
pub fn text(text: impl Into<String>) -> Exhibit {
    Exhibit::Text(text.into())
}

/// A chart.
pub fn plot(plot: Plot<'static>) -> Exhibit {
    Exhibit::Plot(Box::new(plot))
}

/// A graph picture.
pub fn graph(graph: Graph) -> Exhibit {
    Exhibit::Graph(graph)
}

/// A notebook card.
pub fn card(html: impl Into<String>) -> Exhibit {
    Exhibit::Card(html.into())
}

/// Several readings under tabs, in order.
pub fn panels<const N: usize>(items: [(&str, Exhibit); N]) -> Exhibit {
    Exhibit::Panels(
        items
            .into_iter()
            .map(|(label, exhibit)| (label.to_string(), exhibit))
            .collect(),
    )
}

/// Joins lines of program output into one text plate.
pub fn lines<const N: usize>(items: [String; N]) -> Exhibit {
    Exhibit::Text(items.join("\n"))
}
