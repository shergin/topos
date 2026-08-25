//! The tape card: the recording so far, as the spec dump.

use malevich::Theme;

use super::html;
use crate::{Element, Tape};

impl<E: Element> Tape<E> {
    /// Renders the tape as a self-contained HTML card: the IR dump
    /// so far, the same text [`describe`](Tape::describe) answers.
    ///
    /// Rendering is pure and deterministic for a given tape and theme.
    pub fn to_html(&self, theme: Theme) -> String {
        let described = self.describe();
        let summary = described
            .lines()
            .last()
            .unwrap_or("tape: 0 nodes")
            .to_string();
        html::card(theme, &html::escape(&summary), &html::dump_pre(&described))
    }

    /// Displays the tape when it is the last expression in an Evcxr
    /// cell.
    pub fn evcxr_display(&self) {
        html::show(&self.to_html(Theme::detect()), &self.describe());
    }
}

#[cfg(test)]
#[path = "tests/tape_tests.rs"]
mod tests;
