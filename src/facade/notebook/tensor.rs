//! The payload card, for a tensor held on its own.
//!
//! A notebook reaches a bare `Tensor` constantly — a batch, a weight
//! read back out of a generation, a running estimate — so the payload
//! renderer is reachable from the type directly and not only through
//! the values that carry it.

use malevich::Theme;

use super::{html, render};
use crate::{Element, Emittable, Tensor};

impl<E: Element + Emittable> Tensor<E>
where
    f64: From<E>,
{
    /// Renders the tensor as a self-contained HTML card: shape, element
    /// type, and extremes, then the values — an exact table while they
    /// are few, a chart once they are many.
    ///
    /// Rendering is pure and deterministic for a given tensor and theme.
    pub fn to_html(&self, theme: Theme) -> String {
        render::payload_card(theme, "tensor", self)
    }

    /// Displays the tensor when it is the last expression in an Evcxr
    /// cell.
    pub fn evcxr_display(&self) {
        html::show(
            &self.to_html(Theme::detect()),
            &render::payload_text("tensor", self),
        );
    }
}

#[cfg(test)]
#[path = "tests/tensor_tests.rs"]
mod tests;
