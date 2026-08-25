//! The Evcxr output protocol and the shared card chrome.
//!
//! Evcxr reads mime-typed blocks from a cell's stdout, so emitting
//! rich output is printing text in a fixed envelope. The envelope and
//! the card background come from `malevich`, which paints its own plot
//! cards with them, so a chart and a tensor table rendered side by side
//! cannot disagree. What is local to this module is the chrome topos
//! adds on top: the header line and the muted color it is drawn in.

use malevich::Theme;
use malevich::evcxr::card_colors;

pub(crate) use malevich::evcxr::mime_bundle;

/// The muted foreground for headers and units, per theme.
pub(crate) fn muted_color(theme: Theme) -> &'static str {
    if theme == Theme::LIGHT {
        "#59636e"
    } else {
        "#8d96a0"
    }
}

/// Escapes text for HTML element content.
pub(crate) fn escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            _ => escaped.push(character),
        }
    }
    escaped
}

/// Longest spec dump placed in a card before the middle is elided.
pub(crate) const DUMP_LINE_LIMIT: usize = 80;

/// The CSS for a spec or schedule `<pre>` block.
const DUMP_PRE_STYLE: &str = "margin:0;white-space:pre;overflow-x:auto";

/// Elides the middle of a multi-line dump so a large spec cannot
/// freeze a notebook cell. The last line (the summary) is kept.
pub(crate) fn truncate_dump(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= DUMP_LINE_LIMIT {
        return text.to_string();
    }
    let keep_head = DUMP_LINE_LIMIT.saturating_sub(1);
    let omitted = lines.len().saturating_sub(keep_head + 1);
    let mut out = String::new();
    for line in lines.iter().take(keep_head) {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(&format!("... {omitted} more lines\n"));
    if let Some(last) = lines.last() {
        out.push_str(last);
    }
    out
}

/// Escapes `text` into a scrollable `<pre>` block, eliding a long dump.
pub(crate) fn dump_pre(text: &str) -> String {
    format!(
        "<pre style=\"{DUMP_PRE_STYLE}\">{}</pre>",
        escape(&truncate_dump(text))
    )
}

/// Wraps `body` in a themed card carrying `header` as its caption.
///
/// The body is placed verbatim, so callers pass either escaped text or
/// their own markup.
pub(crate) fn card(theme: Theme, header: &str, body: &str) -> String {
    use std::fmt::Write as _;

    let (background, foreground) = card_colors(theme);
    let muted = muted_color(theme);
    let mut html = String::with_capacity(body.len() + 512);
    let _ = write!(
        html,
        "<div style=\"margin:0;padding:12px 16px;border:0;border-radius:8px;\
         box-sizing:border-box;display:inline-block;max-width:100%;overflow-x:auto;\
         font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;\
         font-size:13px;line-height:1.35;background-color:{background};color:{foreground}\">\
         <div style=\"color:{muted};margin-bottom:6px\">{header}</div>{body}</div>"
    );
    html
}

/// Prints an HTML representation and its plain-text alternative as one
/// Evcxr value: the last statement of every `evcxr_display`.
pub(crate) fn show(html: &str, plain: &str) {
    println!(
        "{}",
        mime_bundle(&[("text/html", html), ("text/plain", plain)])
    );
}

#[cfg(test)]
#[path = "tests/html_tests.rs"]
mod tests;
