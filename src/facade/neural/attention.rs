//! Scaled dot-product attention: the score-softmax-value formula as
//! a named composition, plus the causal-mask payload factory.
//!
//! Attention is a formula, not a primitive: one head is four
//! recorded operations around the stable softmax composition, so its
//! gradient falls out of the chain rule with no dedicated backward
//! rule, and pattern matching stays provenance-blind — a hand-rolled
//! spelling behaves identically. Projections, head loops, rotary
//! embeddings, grouped queries, and caches are architecture, not
//! attention; they stay with the caller, because fused-QKV and
//! separate-projection checkpoints split heads differently and a
//! facade that picked one would fight the other.

use crate::{Element, Tensor, Value};

/// Records one-head scaled dot-product attention:
/// `softmax((query @ key^T) * scale + mask) @ value`.
///
/// This is the spelling of record — the exact node sequence the
/// in-repo transformers hand-rolled before the facade existed — so a
/// migration from the hand-rolled form is bit-identical.
///
/// # Parameters
/// - `query`: the `[tokens, dim]` queries of one head.
/// - `key`, `value`: the `[source, dim]` rows attended over — the
///   same rows as the queries for self-attention, cached rows for
///   decode.
/// - `mask`: `[tokens, source]`, additive: zero where attention is
///   allowed, a most-negative fill where it is not.
///   [`causal_mask`] mints the standard triangular payload.
/// - `scale`: rank 0, broadcast over the scores. The caller owns
///   the constant (`1 / sqrt(dim)` in the standard recipe): facades
///   never choose float constants.
///
/// The head loop stays with the caller: grouped queries, fused QKV
/// projections, and one-row decode each split heads differently.
///
/// # Panics
/// Panics if the values belong to different networks, or as the
/// recorded operations panic on rank or axis disagreement.
pub fn scaled_dot_product<'tape, E: Element>(
    query: Value<'tape, E>,
    key: Value<'tape, E>,
    value: Value<'tape, E>,
    mask: Value<'tape, E>,
    scale: Value<'tape, E>,
) -> Value<'tape, E> {
    let scores = query.matmul(key.transpose());
    let weights = (scores * scale.broadcast_like(scores) + mask).softmax(1);
    weights.matmul(value)
}

/// Returns the additive causal mask payload of shape
/// `[extent, extent]`: zero on and below the diagonal, `fill`
/// strictly above, so position `i` may attend to positions `0..=i`.
///
/// Host-side, like the `init` factories: it mints a payload the
/// caller records as a leaf. The caller supplies `fill` — the
/// standard choice is the element's negative infinity, converted at
/// the caller's own boundary — because a facade never chooses float
/// constants, and a custom element may not have an infinity to
/// choose.
pub fn causal_mask<E: Element>(extent: usize, fill: E) -> Tensor<E> {
    let mut elements = Vec::with_capacity(extent * extent);
    for query in 0..extent {
        for key in 0..extent {
            elements.push(if key <= query {
                E::zero()
            } else {
                fill.clone()
            });
        }
    }
    Tensor::new([extent, extent], elements)
}

#[cfg(test)]
#[path = "tests/attention_tests.rs"]
mod tests;
