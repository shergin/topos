//! The parameters card: slot count, then each slot's shape.

use malevich::Theme;

use super::{html, render};
use crate::{Element, Parameters};

/// Most slots listed before the middle of the table is elided.
const SLOT_LIMIT: usize = 24;

impl<E: Element> Parameters<E>
where
    f64: From<E>,
{
    /// Renders the state as a self-contained HTML card: how many
    /// slots, then each slot's shape — and the value when it is a
    /// scalar.
    ///
    /// Rendering is pure and deterministic for a given state and
    /// theme.
    pub fn to_html(&self, theme: Theme) -> String {
        html::card(theme, &html::escape(&self.summary()), &self.slots_html())
    }

    /// Displays the state when it is the last expression in an Evcxr
    /// cell.
    pub fn evcxr_display(&self) {
        html::show(&self.to_html(Theme::detect()), &self.slots_text());
    }

    fn summary(&self) -> String {
        let slots = self.len();
        let plural = if slots == 1 { "" } else { "s" };
        format!("parameters  \u{b7}  {slots} slot{plural}")
    }

    fn slot_lines(&self) -> Vec<String> {
        self.payloads()
            .iter()
            .enumerate()
            .map(|(index, payload)| {
                let shape = render::shape_text(&payload.shape());
                if payload.shape().volume() == 1 {
                    let value = match payload.as_constant() {
                        Some(element) => f64::from(element.clone()),
                        None => payload.iter().next().map(f64::from).unwrap_or(0.0),
                    };
                    format!("{index:>4}  {shape}  {}", render::number(value))
                } else {
                    format!("{index:>4}  {shape}")
                }
            })
            .collect()
    }

    fn listed_lines(&self) -> String {
        let lines = self.slot_lines();
        if lines.len() <= SLOT_LIMIT {
            return lines.join("\n");
        }
        let keep = SLOT_LIMIT.saturating_sub(1);
        let omitted = lines.len() - keep;
        let mut out = lines[..keep].join("\n");
        out.push('\n');
        out.push_str(&format!("... {omitted} more slots"));
        out
    }

    fn slots_html(&self) -> String {
        if self.is_empty() {
            return "<div>no parameters</div>".to_string();
        }
        html::dump_pre(&self.listed_lines())
    }

    fn slots_text(&self) -> String {
        format!("{}\n{}", self.summary(), self.listed_lines())
    }
}

#[cfg(test)]
#[path = "tests/parameters_tests.rs"]
mod tests;
