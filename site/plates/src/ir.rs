//! The IR dump as HTML: the exact text `describe` prints, with every
//! column marked so the page can color opcodes, link operand
//! references to the lines they name, and shade the lines a
//! transform appended.
//!
//! The text is copied character for character — whitespace included,
//! so the columns stay aligned — and only the tokens are wrapped.

use crate::escape;

/// A dump and the way it should be shown.
pub struct Dump<'a> {
    /// The text, as `describe` printed it.
    pub text: &'a str,
    /// The first node index of an appended region, and the divider
    /// label drawn above it — a gradient recorded by `differentiate`,
    /// nodes recorded after a reopen.
    pub split: Option<(usize, &'a str)>,
    /// Node indices to underline as the ones the reader should find:
    /// a plan's results, a declared observe.
    pub marks: &'a [usize],
}

impl<'a> Dump<'a> {
    /// The plain form: every line as `describe` wrote it.
    pub fn new(text: &'a str) -> Self {
        Dump {
            text,
            split: None,
            marks: &[],
        }
    }
}

/// Opcodes that are sources: supplied, not computed.
const SOURCES: [&str; 3] = ["Leaf", "Parameter", "Input"];

/// The words a plan's liveness column uses.
const LIVENESS: [&str; 5] = ["kept", "fused", "retained", "freed", "releasable"];

/// Renders a dump as the inside of a `<pre class="ir">`.
pub fn to_html(dump: &Dump<'_>) -> String {
    let mut out = String::with_capacity(dump.text.len() * 3);
    let mut divided = false;
    for line in dump.text.lines() {
        if let Some((split, label)) = dump.split
            && !divided
            && node_index(line).is_some_and(|index| index >= split)
        {
            divided = true;
            out.push_str(&format!(
                "<span class=\"ir-divider\">{}</span>\n",
                escape(label)
            ));
        }
        out.push_str(&line_html(line, dump, divided));
        out.push('\n');
    }
    out
}

/// The node index a dump line starts with, if it is a node line.
fn node_index(line: &str) -> Option<usize> {
    let trimmed = line.trim_start();
    let digits: String = trimmed.chars().take_while(char::is_ascii_digit).collect();
    if digits.is_empty() {
        return None;
    }
    let rest = &trimmed[digits.len()..];
    if !rest.starts_with(' ') {
        return None;
    }
    digits.parse().ok()
}

fn line_html(line: &str, dump: &Dump<'_>, derived: bool) -> String {
    let Some(index) = node_index(line) else {
        if line.trim().is_empty() {
            return String::new();
        }
        return format!("<span class=\"ir-summary\">{}</span>", escape(line));
    };
    let mut classes = vec!["ir-line"];
    if derived {
        classes.push("ir-derived");
    }
    if dump.marks.contains(&index) {
        classes.push("ir-marked");
    }
    let mut out = format!(
        "<span class=\"{}\" data-node=\"{index}\">",
        classes.join(" ")
    );
    let chars: Vec<char> = line.chars().collect();
    let mut at = 0;
    // The index column.
    let leading = chars.iter().take_while(|c| **c == ' ').count();
    out.push_str(&" ".repeat(leading));
    at += leading;
    let digits_end = at
        + chars[at..]
            .iter()
            .take_while(|c| c.is_ascii_digit())
            .count();
    out.push_str(&format!(
        "<span class=\"ir-index\">{}</span>",
        chars[at..digits_end].iter().collect::<String>()
    ));
    at = digits_end;
    // The opcode column.
    let spaces = chars[at..].iter().take_while(|c| **c == ' ').count();
    out.push_str(&" ".repeat(spaces));
    at += spaces;
    let name_end = at + chars[at..].iter().take_while(|c| **c != ' ').count();
    let name: String = chars[at..name_end].iter().collect();
    let class = if SOURCES.contains(&name.as_str()) {
        "ir-op ir-source"
    } else {
        "ir-op"
    };
    out.push_str(&format!("<span class=\"{class}\">{}</span>", escape(&name)));
    at = name_end;
    // The rest: operand references, attributes, the shape, liveness.
    let mut seen_attribute = false;
    let mut seen_shape = false;
    while at < chars.len() {
        let spaces = chars[at..].iter().take_while(|c| **c == ' ').count();
        out.push_str(&" ".repeat(spaces));
        at += spaces;
        if at >= chars.len() {
            break;
        }
        // One token: up to the next space, except that a bracket group
        // stays together with what precedes it (`shape=[1, 2]`).
        let mut end = at;
        let mut depth = 0usize;
        while end < chars.len() {
            match chars[end] {
                '[' => depth += 1,
                ']' => depth = depth.saturating_sub(1),
                ' ' if depth == 0 => break,
                _ => {}
            }
            end += 1;
        }
        let token: String = chars[at..end].iter().collect();
        at = end;
        if seen_shape {
            // Everything after the shape is the liveness column; keep
            // it as one span with its own spacing.
            let rest: String = chars[at..].iter().collect();
            let word = if LIVENESS.contains(&token.as_str()) {
                "ir-live"
            } else {
                "ir-note"
            };
            out.push_str(&format!(
                "<span class=\"{word}\">{}{}</span>",
                escape(&token),
                escape(&rest)
            ));
            break;
        }
        if token.starts_with('[') {
            seen_shape = true;
            out.push_str(&format!(
                "<span class=\"ir-shape\">{}</span>",
                escape(&token)
            ));
            continue;
        }
        if token.contains('=') {
            seen_attribute = true;
            out.push_str(&format!(
                "<span class=\"ir-attr\">{}</span>",
                escape(&token)
            ));
            continue;
        }
        let (digits, comma) = match token.strip_suffix(',') {
            Some(digits) => (digits, ","),
            None => (token.as_str(), ""),
        };
        if !seen_attribute && !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
            out.push_str(&format!(
                "<span class=\"ir-ref\" data-ref=\"{digits}\">{digits}</span>{comma}"
            ));
            continue;
        }
        out.push_str(&escape(&token));
    }
    out.push_str("</span>");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_node_line_marks_every_column() {
        let html = to_html(&Dump::new(
            "   7  Broadcast      6  shape=[1, 2]    [1, 2]\n",
        ));
        assert!(html.contains("data-node=\"7\""));
        assert!(html.contains("<span class=\"ir-ref\" data-ref=\"6\">6</span>"));
        assert!(html.contains("<span class=\"ir-attr\">shape=[1, 2]</span>"));
        assert!(html.contains("<span class=\"ir-shape\">[1, 2]</span>"));
    }

    #[test]
    fn a_plan_line_keeps_its_liveness() {
        let html = to_html(&Dump::new(
            "   2  MatMul         1, 0               [1, 2]       freed after 3\n",
        ));
        assert!(html.contains("<span class=\"ir-live\">freed after 3</span>"));
        assert!(html.contains("data-ref=\"1\""));
        assert!(html.contains("data-ref=\"0\""));
    }

    #[test]
    fn a_summary_line_is_one_span() {
        let html = to_html(&Dump::new("tape: 6 nodes, 1 parameter, 1 input\n"));
        assert_eq!(
            html,
            "<span class=\"ir-summary\">tape: 6 nodes, 1 parameter, 1 input</span>\n"
        );
    }

    #[test]
    fn the_split_draws_a_divider_once() {
        let text = "   0  Parameter                         [2, 2]\n   1  Leaf                              []\n   2  Add            1, 0               [2, 2]\n";
        let html = to_html(&Dump {
            text,
            split: Some((1, "appended")),
            marks: &[],
        });
        assert_eq!(html.matches("ir-divider").count(), 1);
        assert_eq!(html.matches("ir-derived").count(), 2);
    }
}
