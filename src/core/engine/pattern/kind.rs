use crate::backend::Formula;
use crate::graph::Symbol;

use super::pattern::Pattern;

/// The closed set of graph patterns the catalog recognizes: the
/// public kind of a [`PatternMatch`].
///
/// It is deliberately not [`Formula`]: the acceleration vocabulary
/// also names `Gemm` and `Map`, which are payload tasks, not graph
/// shapes. Two closed enums are the honest split;
/// [`PatternMatch::formula`] maps a graph shape onto its vocabulary
/// entry. Adding a pattern is a new variant plus a matcher — a
/// visible, breaking change at every match site, like an opcode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternKind {
    /// The canonical im2col chain feeding a rank-2 `matmul`.
    WindowProduct,
    /// The canonical max-pool window fold ending in the facade
    /// squeeze.
    ReduceWindow,
    /// Batch normalization by the batch's own statistics, with the
    /// mean and variance as named results.
    BatchNormTraining,
    /// Batch normalization by supplied statistics.
    BatchNormInference,
}

/// One recognized pattern as data: its kind, the root node whose
/// result is the group's result, and every claimed node.
///
/// A match is a compile-time recognition over frozen structure,
/// never a tape rewrite; what a consumer does with one is that
/// consumer's action table. [`Plan::candidates`](crate::Plan::candidates)
/// answers the discovered pool and
/// [`Plan::patterns`](crate::Plan::patterns) the home run's elected
/// groups, so what `describe` prints as "fused" is reconstructible
/// as data.
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub(crate) kind: PatternKind,
    pub(crate) root: Symbol,
    pub(crate) nodes: Vec<Symbol>,
}

impl PatternMatch {
    /// Returns which pattern matched.
    pub fn kind(&self) -> PatternKind {
        self.kind
    }

    /// Returns the root node: the node whose result is the group's.
    pub fn root(&self) -> Symbol {
        self.root
    }

    /// Returns every node the pattern claims — interiors, named
    /// results, and the root — in allocation order.
    pub fn nodes(&self) -> &[Symbol] {
        &self.nodes
    }

    /// Returns the acceleration-vocabulary entry this pattern is the
    /// graph face of.
    pub fn formula(&self) -> Formula {
        match self.kind {
            PatternKind::WindowProduct => Formula::WindowProduct,
            PatternKind::ReduceWindow => Formula::ReduceWindow,
            PatternKind::BatchNormTraining => Formula::BatchNormTraining,
            PatternKind::BatchNormInference => Formula::BatchNormInference,
        }
    }
}

impl Pattern {
    /// Returns the public kind of this internal match.
    pub(crate) fn kind(&self) -> PatternKind {
        match self {
            Pattern::WindowProduct(_) => PatternKind::WindowProduct,
            Pattern::ReduceWindow(_) => PatternKind::ReduceWindow,
            Pattern::BatchNormTraining(_) => PatternKind::BatchNormTraining,
            Pattern::BatchNormInference(_) => PatternKind::BatchNormInference,
        }
    }
}
