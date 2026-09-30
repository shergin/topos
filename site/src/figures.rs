//! The figures: blocks of Rust in `site/figures/*.rs`, each recording
//! a graph and asking the crate for a reading of it, rendered at
//! build time. A page asks for one with a `{{figure name}}` line; the
//! registry renders the reading and shows beside it the exact code
//! that ran — the block is both `include!`d and `include_str!`ed, so
//! the two cannot drift.

use std::cell::Cell;

use malevich::render::{Charset, ColorMode};
use malevich::{Frame, Theme};
use topos_plates::escape;
use topos_plates::graph;
use topos_plates::highlight::highlight;
use topos_plates::ir::{self, Dump};

use crate::markdown::Context;
use crate::show::Exhibit;

/// One figure: a reading, the frame its charts are meant for, and
/// its source.
pub struct Figure {
    pub name: &'static str,
    pub caption: &'static str,
    pub frame: Frame,
    pub source: &'static str,
    pub exhibit: Exhibit,
}

/// Every figure, by name.
pub struct Registry {
    figures: Vec<Figure>,
    rendered: Cell<usize>,
}

/// A frame for the chart cards: quadrants, truecolor, the dark card.
pub fn card(width: usize, height: usize) -> Frame {
    Frame {
        width,
        height,
        charset: Charset::Quadrants,
        color: ColorMode::TrueColor,
        theme: Theme::DARK,
    }
}

macro_rules! catalog {
    ($($name:literal => ($caption:literal, $frame:expr)),* $(,)?) => {
        vec![$(Figure {
            name: $name,
            caption: $caption,
            frame: $frame,
            source: include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/figures/", $name, ".rs")),
            exhibit: include!(concat!(env!("CARGO_MANIFEST_DIR"), "/figures/", $name, ".rs")),
        }),*]
    };
}

mod catalog;

impl Registry {
    pub fn new() -> Registry {
        Registry {
            figures: catalog::all(),
            rendered: Cell::new(0),
        }
    }

    /// How many plates were rendered so far.
    pub fn rendered(&self) -> usize {
        self.rendered.get()
    }

    /// Every figure's name, in catalog order.
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.figures.iter().map(|figure| figure.name)
    }

    fn figure(&self, name: &str) -> Result<&Figure, String> {
        self.figures
            .iter()
            .find(|figure| figure.name == name)
            .ok_or_else(|| format!("no figure named {name}"))
    }

    /// Expands one `{{kind arguments | caption}}` directive.
    pub fn directive(
        &self,
        kind: &str,
        arguments: &[&str],
        caption: Option<&str>,
        context: &Context,
        plates: &Cell<usize>,
        scripts: &mut Vec<String>,
    ) -> Result<String, String> {
        let next = || {
            plates.set(plates.get() + 1);
            plates.get()
        };
        let first = || {
            arguments
                .first()
                .copied()
                .ok_or_else(|| "a figure name is required".to_string())
        };
        let caption_of = |figure: &Figure| caption.unwrap_or(figure.caption).to_string();
        match kind {
            // The reading, with the code that produced it.
            "figure" => {
                let figure = self.figure(first()?)?;
                let show_code = !arguments.contains(&"nocode");
                let inner = self.render(&figure.exhibit, &figure.frame, figure.caption);
                Ok(plate(
                    next(),
                    &inner,
                    &caption_of(figure),
                    show_code.then_some(figure.source),
                    class_of(&figure.exhibit),
                ))
            }
            // The code of a figure, without rendering it.
            "code" => {
                let figure = self.figure(first()?)?;
                Ok(crate::markdown::code_block("rust", &dedent(figure.source)))
            }
            // Two or more chart cards side by side.
            "pair" => {
                let mut inner = String::from("<div class=\"pair\">");
                let mut captions = Vec::new();
                for name in arguments {
                    let figure = self.figure(name)?;
                    inner.push_str(&self.render(&figure.exhibit, &figure.frame, figure.caption));
                    captions.push(figure.caption);
                }
                inner.push_str("</div>");
                let caption = caption
                    .map(str::to_string)
                    .unwrap_or_else(|| captions.join(" "));
                Ok(plate(next(), &inner, &caption, None, "pair-plate"))
            }
            // A gallery example inline: its recorded output and source.
            "gallery" => {
                let name = first()?;
                let entry = crate::gallery::entry(name)
                    .ok_or_else(|| format!("no gallery example named {name}"))?;
                Ok(crate::gallery::entry_html(context, entry, next()))
            }
            // The playground, driven by the wasm build.
            "playground" => {
                scripts.push("playground.js".to_string());
                Ok(playground(next()))
            }
            other => Err(format!("unknown directive {other}")),
        }
    }

    /// Renders a reading as HTML.
    pub fn render(&self, exhibit: &Exhibit, frame: &Frame, title: &str) -> String {
        self.rendered.set(self.rendered.get() + 1);
        match exhibit {
            Exhibit::Spec { text, split, marks } => {
                let dump = Dump {
                    text,
                    split: split.as_ref().map(|(at, label)| (*at, label.as_str())),
                    marks,
                };
                format!("<pre class=\"term ir\">{}</pre>", ir::to_html(&dump))
            }
            Exhibit::Hlo(text) => format!(
                "<div class=\"code hlo\" data-lang=\"StableHLO\"><pre><code>{}</code></pre></div>",
                highlight("mlir", text.trim_end())
            ),
            Exhibit::Text(text) => {
                format!("<pre class=\"term\">{}</pre>", escape(text.trim_end()))
            }
            Exhibit::Plot(plot) => svg_card(plot, frame, title),
            Exhibit::Graph(graph) => {
                format!(
                    "<div class=\"graph-plate\">{}</div>",
                    graph::svg(graph, title)
                )
            }
            Exhibit::Card(html) => format!("<div class=\"html-card\">{html}</div>"),
            Exhibit::Panels(items) => {
                let mut tabs = String::from("<div class=\"tabs\" role=\"tablist\">");
                let mut panels = String::new();
                for (index, (label, inner)) in items.iter().enumerate() {
                    let selected = index == 0;
                    tabs.push_str(&format!(
                        "<button type=\"button\" role=\"tab\" aria-selected=\"{selected}\" data-panel=\"{index}\">{}</button>",
                        escape(label)
                    ));
                    panels.push_str(&format!(
                        "<div class=\"panel\" role=\"tabpanel\" data-panel=\"{index}\"{}>{}</div>",
                        if selected { "" } else { " hidden" },
                        self.render(inner, frame, title)
                    ));
                }
                tabs.push_str("</div>");
                format!("<div class=\"panels\">{tabs}{panels}</div>")
            }
        }
    }
}

/// The SVG card of a chart in a frame, sized by CSS rather than by its
/// own width and height, with an accessible name.
fn svg_card(plot: &malevich::Plot<'static>, frame: &Frame, title: &str) -> String {
    let svg = plot.to_svg(frame);
    let head_end = svg.find('>').expect("svg root element");
    let head = &svg[..head_end];
    let rest = &svg[head_end..];
    let mut attributes = String::new();
    for attribute in head.split_whitespace().skip(1) {
        if attribute.starts_with("width=") || attribute.starts_with("height=") {
            continue;
        }
        attributes.push(' ');
        attributes.push_str(attribute);
    }
    format!(
        "<svg class=\"card\" role=\"img\" aria-label=\"{}\" style=\"max-width:{}px\"{attributes}{rest}",
        escape(title),
        32.0 + frame.width as f64 * 7.8,
    )
}

fn class_of(exhibit: &Exhibit) -> &'static str {
    match exhibit {
        Exhibit::Spec { .. } => "ir-plate",
        Exhibit::Hlo(_) => "hlo-plate",
        Exhibit::Text(_) => "text",
        Exhibit::Plot(_) => "card",
        Exhibit::Graph(_) => "graph-figure",
        Exhibit::Card(_) => "card-html",
        Exhibit::Panels(_) => "panels-plate",
    }
}

/// Strips the outer braces and one indent level from a figure's block.
pub fn dedent(source: &str) -> String {
    let inner: Vec<&str> = source
        .lines()
        .skip_while(|line| line.trim() != "{")
        .skip(1)
        .collect();
    let end = inner
        .iter()
        .rposition(|line| line.trim() == "}")
        .unwrap_or(inner.len());
    inner[..end]
        .iter()
        .map(|line| line.strip_prefix("    ").unwrap_or(line))
        .collect::<Vec<_>>()
        .join("\n")
        .trim_end()
        .to_string()
}

/// A numbered plate: the rendered reading, its caption, and optionally
/// the code.
fn plate(number: usize, inner: &str, caption: &str, code: Option<&str>, class: &str) -> String {
    let mut out = format!(
        "<figure class=\"plate {class}\">\n{inner}\n<figcaption><span class=\"plate-number\">Plate {number}.</span> {}</figcaption>\n",
        inline_markdown(caption)
    );
    if let Some(code) = code {
        out.push_str(&format!(
            "<details class=\"source\"><summary>The code that produced it</summary><div class=\"code\" data-lang=\"Rust\"><pre><code>{}</code></pre></div></details>\n",
            highlight("rust", &dedent(code))
        ));
    }
    out.push_str("</figure>\n");
    out
}

/// Backticks to `<code>` and nothing else — captions stay one sentence.
pub fn inline_markdown(text: &str) -> String {
    let mut out = String::new();
    let mut code = false;
    for part in text.split('`') {
        if code {
            out.push_str("<code>");
            out.push_str(&escape(part));
            out.push_str("</code>");
        } else {
            out.push_str(&escape(part));
        }
        code = !code;
    }
    out
}

fn playground(number: usize) -> String {
    format!(
        "<figure class=\"plate playground-plate\" id=\"playground\">\n\
         <div class=\"playground\">\n\
         <div class=\"editor\">\n\
         <div class=\"samples\" id=\"samples\"></div>\n\
         <textarea id=\"program\" spellcheck=\"false\" aria-label=\"The program\" rows=\"12\"></textarea>\n\
         <p class=\"error\" id=\"error\" hidden></p>\n\
         <div class=\"knobs\">\n\
         <label>Differentiate <select id=\"target\" aria-label=\"Target\"></select></label>\n\
         <label>Observe <select id=\"observe\" aria-label=\"Observe\"></select></label>\n\
         <label class=\"check\"><input type=\"checkbox\" id=\"exact\"> Exact numerics</label>\n\
         </div>\n\
         </div>\n\
         <div class=\"readings\" id=\"readings\"><p class=\"status\">Loading the engine…</p></div>\n\
         </div>\n\
         <figcaption><span class=\"plate-number\">Plate {number}.</span> The program on the left is recorded by the crate compiled to WebAssembly, and every tab on the right is one reading of that recording, remade as you type. Nothing is rewritten between the tabs; they are the same graph.</figcaption>\n\
         </figure>\n"
    )
}
