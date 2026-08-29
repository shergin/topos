//! The entry card: the declared reading.

use malevich::Theme;

use super::html;
use crate::{BoundEntry, Element, Entry, Numerics, Symbol};

impl Entry {
    /// Renders the declared reading: roots, observes, memory
    /// posture, numerics.
    pub fn to_html(&self, theme: Theme) -> String {
        html::card(
            theme,
            &html::escape(&self.summary()),
            &html::dump_pre(&self.body_text()),
        )
    }

    /// Displays the entry when it is the last expression in an
    /// Evcxr cell.
    pub fn evcxr_display(&self) {
        html::show(
            &self.to_html(Theme::detect()),
            &format!("{}\n{}", self.summary(), self.body_text()),
        );
    }

    fn summary(&self) -> String {
        let roots = self.roots.len();
        let observes = self.observe.len();
        let numerics = match self.numerics {
            Numerics::Fast => "Fast",
            Numerics::Exact => "Exact",
        };
        format!("entry  \u{b7}  {roots} roots  \u{b7}  {observes} observes  \u{b7}  {numerics}")
    }

    fn body_text(&self) -> String {
        let roots = list_symbols(&self.roots);
        let observes = if self.observe.is_empty() {
            "(none)".to_string()
        } else {
            list_symbols(&self.observe)
        };
        let memory = if self.backward {
            "retain for backward"
        } else {
            "forward-only"
        };
        format!("roots     {roots}\nobserves  {observes}\nmemory    {memory}")
    }
}

fn list_symbols(symbols: &[Symbol]) -> String {
    symbols
        .iter()
        .map(|symbol| format!("#{}", symbol.index()))
        .collect::<Vec<_>>()
        .join("  ")
}

impl<E: Element> BoundEntry<'_, E> {
    /// Renders the bound entry the same way as its detached
    /// signature.
    pub fn to_html(&self, theme: Theme) -> String {
        self.entry().to_html(theme)
    }

    /// Displays the bound entry when it is the last expression in
    /// an Evcxr cell.
    pub fn evcxr_display(&self) {
        self.entry().evcxr_display();
    }
}

#[cfg(test)]
#[path = "tests/entry_tests.rs"]
mod tests;
