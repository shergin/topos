//! The network card: the sealed spec's one-line summary.

use malevich::Theme;

use super::html;
use crate::{Element, Network};

impl<E: Element> Network<E> {
    /// Renders the network as a self-contained HTML card: the IR
    /// dump, one line per recorded node — the same text
    /// [`describe`](Network::describe) answers, so the notebook and
    /// the terminal cannot disagree.
    ///
    /// Rendering is pure and deterministic for a given network and
    /// theme, which is what makes it testable.
    pub fn to_html(&self, theme: Theme) -> String {
        let described = self.describe();
        let summary = described
            .lines()
            .last()
            .unwrap_or("network: 0 nodes")
            .to_string();
        html::card(theme, &html::escape(&summary), &html::dump_pre(&described))
    }

    /// Displays the network when it is the last expression in an
    /// Evcxr cell.
    pub fn evcxr_display(&self) {
        html::show(&self.to_html(Theme::detect()), &self.describe());
    }
}

#[cfg(test)]
#[path = "tests/network_tests.rs"]
mod tests;
