//! Cards for the two ways to designate a value, and for a finished run.

use malevich::Theme;

use super::{html, render};
use crate::{Element, Emittable, Run, Symbol, Value};

impl<E: Element + Emittable> Value<'_, E>
where
    f64: From<E>,
{
    /// Renders the proxy's stored payload as a self-contained HTML
    /// card.
    ///
    /// The payload shown is the recorded one: a leaf's constant, a
    /// parameter's record-site initial, or an input's default. Live
    /// parameter payloads belong to the caller's
    /// [`Parameters`](crate::Parameters) and are read by [`Symbol`].
    pub fn to_html(&self, theme: Theme) -> String {
        let Some(payload) = self.payload() else {
            let header = format!(
                "value  \u{b7}  {}  \u{b7}  not yet computed",
                render::shape_text(&self.shape())
            );
            return html::card(
                theme,
                &html::escape(&header),
                "<div>run <code>forward</code> to give this value a payload</div>",
            );
        };
        render::payload_card(theme, "value", &payload)
    }

    /// Displays the value when it is the last expression in an Evcxr
    /// cell.
    pub fn evcxr_display(&self) {
        let plain = match self.payload() {
            Some(payload) => render::payload_text("value", &payload),
            None => format!(
                "value  {}  not yet computed",
                render::shape_text(&self.shape())
            ),
        };
        html::show(&self.to_html(Theme::detect()), &plain);
    }
}

impl Symbol {
    /// Renders the symbol as a self-contained HTML card.
    ///
    /// A symbol is a detached name and carries no payload of its own,
    /// so the card says what it is and how to read through it rather
    /// than inventing a value.
    pub fn to_html(&self, theme: Theme) -> String {
        html::card(
            theme,
            "symbol",
            "<div>a detached name; <code>parameters.of(symbol)</code> and \
             <code>run.of(symbol)</code> read through it, and \
             <code>tape.resolve(symbol)</code> reenters recording</div>",
        )
    }

    /// Displays the symbol when it is the last expression in an Evcxr
    /// cell.
    pub fn evcxr_display(&self) {
        html::show(
            &self.to_html(Theme::detect()),
            "symbol  \u{b7}  a detached name; read through parameters, runs, and fields",
        );
    }
}

impl<E: Element> Run<E>
where
    f64: From<E>,
{
    /// Renders the run as a self-contained HTML card.
    ///
    /// A run holds a value per node, so the card shows the profile of
    /// their magnitudes: the shape of the forward pass, where it grew,
    /// and whether anything went non-finite.
    pub fn to_html(&self, theme: Theme) -> String {
        super::field::profile_card(theme, "run", self.field())
    }

    /// Displays the run when it is the last expression in an Evcxr
    /// cell.
    pub fn evcxr_display(&self) {
        html::show(
            &self.to_html(Theme::detect()),
            &super::field::profile_text("run", self.field()),
        );
    }
}

#[cfg(test)]
#[path = "tests/value_tests.rs"]
mod tests;
