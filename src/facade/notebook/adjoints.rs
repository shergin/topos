//! The adjoints card: the recorded reverse-mode pairing.

use malevich::Theme;

use super::html;
use crate::Adjoints;

impl Adjoints {
    /// Renders the pairing as a self-contained HTML card: the
    /// target, then each `wrt → gradient` pair in `wrt` order.
    pub fn to_html(&self, theme: Theme) -> String {
        html::card(
            theme,
            &html::escape(&self.summary()),
            &html::dump_pre(&self.body_text()),
        )
    }

    /// Displays the adjoints when they are the last expression in
    /// an Evcxr cell.
    pub fn evcxr_display(&self) {
        html::show(
            &self.to_html(Theme::detect()),
            &format!("{}\n{}", self.summary(), self.body_text()),
        );
    }

    fn summary(&self) -> String {
        let pairs = self.pairs().len();
        let plural = if pairs == 1 { "" } else { "s" };
        format!(
            "adjoints  \u{b7}  {pairs} pair{plural}  \u{b7}  target #{}",
            self.target().id.index()
        )
    }

    fn body_text(&self) -> String {
        self.pairs()
            .iter()
            .map(|&(wrt, gradient)| format!("#{}  →  #{}", wrt.id.index(), gradient.id.index()))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
#[path = "tests/adjoints_tests.rs"]
mod tests;
