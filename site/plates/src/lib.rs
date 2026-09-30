//! The plates the site and the playground share: a spec dump as
//! HTML with every operand reference marked, a graph as a layered
//! SVG picture, and code as highlighted HTML.
//!
//! Nothing here draws anything by hand. The IR text comes from
//! `describe`, the graph from `Network::nodes`, and both are turned
//! into markup the page styles.

pub mod graph;
pub mod highlight;
pub mod ir;

/// Escapes text for HTML element content and attribute values.
pub fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            c => out.push(c),
        }
    }
    out
}
