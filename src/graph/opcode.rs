use std::fmt::Write;

use smallvec::SmallVec;

use crate::{MapOperation, Shape};

use super::Symbol;

/// The public opcode of one recorded node: the payload-free twin of
/// the engine's operation enum.
///
/// It is a closed set on purpose — the op vocabulary is the IR, and
/// adding a variant is a breaking IR change that should fail to
/// compile at every match site, including user matches on the
/// compiler surface. Variants that take parameters carry them; the
/// rest are tags. A leaf's payload is deliberately absent: the IR
/// view is structure, and payloads of sources are read through
/// [`Network::payload`](crate::Network::payload) or
/// [`Value::payload`](crate::Value::payload).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Opcode {
    /// A constant supplied at recording time.
    Leaf,
    /// A trainable leaf, read from [`Parameters`](crate::Parameters)
    /// per run.
    Parameter,
    /// A declared per-run leaf, overlaid by feeds.
    Input,
    /// Elementwise addition.
    Add,
    /// Elementwise subtraction; kept as one node although `Add` of
    /// `Neg` is bit-exact — a practical decision for one-line specs
    /// and a one-pass oracle.
    Sub,
    /// Elementwise multiplication.
    Mul,
    /// Elementwise division.
    Div,
    /// Elementwise negation.
    Neg,
    /// One unary elementwise transcendental.
    Map {
        /// Which transcendental this node applies.
        operation: MapOperation,
    },
    /// Elementwise power.
    Powf,
    /// Elementwise maximum; ties route the gradient left.
    Maximum,
    /// The Heaviside 0/1 indicator of `operand >= threshold`.
    Step,
    /// The matrix product; ranks above two multiply batched.
    MatMul,
    /// The sum of every element, shaped rank 0.
    Sum,
    /// One axis reduced by summation.
    SumAlong {
        /// The reduced axis.
        axis: usize,
    },
    /// A single element spread across the carried target shape.
    Broadcast {
        /// The target shape.
        shape: Shape,
    },
    /// The operand repeated along one new axis of the carried extent.
    BroadcastAlong {
        /// The inserted axis.
        axis: usize,
        /// The inserted axis's extent.
        extent: usize,
    },
    /// The elements reinterpreted with a new shape in logical order.
    Reshape {
        /// The target shape.
        shape: Shape,
    },
    /// The axes reordered.
    Permute {
        /// Axis `i` of the result takes axis `order[i]` of the operand.
        order: SmallVec<[usize; 4]>,
    },
    /// A window of one axis.
    Narrow {
        /// The narrowed axis.
        axis: usize,
        /// The window's first position.
        start: usize,
        /// The window's extent.
        len: usize,
    },
    /// The operand placed into zeros along one widened axis.
    Pad {
        /// The widened axis.
        axis: usize,
        /// Where the operand's window begins.
        start: usize,
        /// The axis extent after padding.
        full_extent: usize,
    },
    /// Sliding windows along one axis, as a `(count, size)` pair.
    Unfold {
        /// The windowed axis.
        axis: usize,
        /// Elements per window.
        size: usize,
        /// Positions between window starts.
        step: usize,
        /// Positions between window elements.
        dilation: usize,
    },
    /// The `(count, size)` window pair folded back onto one axis.
    Fold {
        /// The window-count axis (`axis + 1` is the size axis).
        axis: usize,
        /// Elements per window.
        size: usize,
        /// Positions between window starts.
        step: usize,
        /// Positions between window elements.
        dilation: usize,
        /// The folded axis extent.
        extent: usize,
    },
    /// Table rows selected by a one-hot selection operand.
    Gather,
    /// Rows scatter-added into a table by a one-hot selection operand;
    /// the result's row count is the selection's vocabulary.
    Scatter,
    /// The stable fused log-softmax along one axis.
    LogSoftmax {
        /// The normalized axis.
        axis: usize,
    },
    /// The stable fused log-sum-exp along one axis.
    LogSumExp {
        /// The reduced axis.
        axis: usize,
    },
}

impl Opcode {
    /// Returns the display name: the same string
    /// [`describe`](crate::Network::describe) prints, with the map
    /// variants named per operation.
    pub fn name(&self) -> &'static str {
        match self {
            Opcode::Leaf => "Leaf",
            Opcode::Parameter => "Parameter",
            Opcode::Input => "Input",
            Opcode::Add => "Add",
            Opcode::Sub => "Sub",
            Opcode::Mul => "Mul",
            Opcode::Div => "Div",
            Opcode::Neg => "Neg",
            Opcode::Map { operation } => match operation {
                MapOperation::Exp => "Exp",
                MapOperation::Ln => "Ln",
                MapOperation::Sqrt => "Sqrt",
                MapOperation::Tanh => "Tanh",
                MapOperation::Sin => "Sin",
                MapOperation::Cos => "Cos",
            },
            Opcode::Powf => "Powf",
            Opcode::Maximum => "Maximum",
            Opcode::Step => "Step",
            Opcode::MatMul => "MatMul",
            Opcode::Sum => "Sum",
            Opcode::SumAlong { .. } => "SumAlong",
            Opcode::Broadcast { .. } => "Broadcast",
            Opcode::BroadcastAlong { .. } => "BroadcastAlong",
            Opcode::Reshape { .. } => "Reshape",
            Opcode::Permute { .. } => "Permute",
            Opcode::Narrow { .. } => "Narrow",
            Opcode::Pad { .. } => "Pad",
            Opcode::Unfold { .. } => "Unfold",
            Opcode::Fold { .. } => "Fold",
            Opcode::Gather => "Gather",
            Opcode::Scatter => "Scatter",
            Opcode::LogSoftmax { .. } => "LogSoftmax",
            Opcode::LogSumExp { .. } => "LogSumExp",
        }
    }

    /// Returns the number of operands this opcode reads.
    pub fn arity(&self) -> usize {
        match self {
            Opcode::Leaf | Opcode::Parameter | Opcode::Input => 0,
            Opcode::Neg
            | Opcode::Map { .. }
            | Opcode::Sum
            | Opcode::SumAlong { .. }
            | Opcode::Broadcast { .. }
            | Opcode::BroadcastAlong { .. }
            | Opcode::Reshape { .. }
            | Opcode::Permute { .. }
            | Opcode::Narrow { .. }
            | Opcode::Pad { .. }
            | Opcode::Unfold { .. }
            | Opcode::Fold { .. }
            | Opcode::LogSoftmax { .. }
            | Opcode::LogSumExp { .. } => 1,
            Opcode::Add
            | Opcode::Sub
            | Opcode::Mul
            | Opcode::Div
            | Opcode::Powf
            | Opcode::Maximum
            | Opcode::Step
            | Opcode::MatMul
            | Opcode::Gather
            | Opcode::Scatter => 2,
        }
    }

    /// Returns whether this opcode is a source: supplied rather than
    /// computed, with no operands.
    pub fn is_source(&self) -> bool {
        matches!(self, Opcode::Leaf | Opcode::Parameter | Opcode::Input)
    }

    /// Renders the parameters a describe line prints after the
    /// operands (`axis=0`, window fields), or an empty string for a
    /// tag variant.
    pub(crate) fn parameter_text(&self) -> String {
        match self {
            Opcode::SumAlong { axis }
            | Opcode::LogSoftmax { axis }
            | Opcode::LogSumExp { axis } => format!("axis={axis}"),
            Opcode::Broadcast { shape } | Opcode::Reshape { shape } => format!("shape={shape}"),
            Opcode::BroadcastAlong { axis, extent } => format!("axis={axis} extent={extent}"),
            Opcode::Permute { order } => {
                let axes: Vec<String> = order.iter().map(usize::to_string).collect();
                format!("order=[{}]", axes.join(", "))
            }
            Opcode::Narrow { axis, start, len } => {
                format!("axis={axis} start={start} len={len}")
            }
            Opcode::Pad {
                axis,
                start,
                full_extent,
            } => format!("axis={axis} start={start} full_extent={full_extent}"),
            Opcode::Unfold {
                axis,
                size,
                step,
                dilation,
            } => format!("axis={axis} size={size} step={step} dilation={dilation}"),
            Opcode::Fold {
                axis,
                size,
                step,
                dilation,
                extent,
            } => format!("axis={axis} size={size} step={step} dilation={dilation} extent={extent}"),
            _ => String::new(),
        }
    }
}

/// One recorded node of the public IR view: a `Copy`-cheap snapshot
/// of the columns — name, operands as [`Symbol`]s, inferred shape —
/// detached from the tape, so a node outlives locks and phases.
///
/// Symbols carry their origin, so a node of one network cannot be
/// silently applied to another.
#[derive(Debug, Clone)]
pub struct Node {
    pub(crate) symbol: Symbol,
    pub(crate) opcode: Opcode,
    pub(crate) shape: Shape,
    pub(crate) operands: SmallVec<[Symbol; 2]>,
}

impl Node {
    /// Returns the name of this node's value.
    pub fn symbol(&self) -> Symbol {
        self.symbol
    }

    /// Returns the operation that produced this node.
    pub fn opcode(&self) -> &Opcode {
        &self.opcode
    }

    /// Returns the opcode's display name.
    pub fn name(&self) -> &'static str {
        self.opcode.name()
    }

    /// Returns the shape inferred when the node was recorded.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the operand names in positional order; empty for
    /// sources.
    pub fn operands(&self) -> &[Symbol] {
        &self.operands
    }

    /// Returns whether this node is a source: supplied, not computed.
    pub fn is_source(&self) -> bool {
        self.opcode.is_source()
    }

    /// Renders this node as one spec line: index, name, operand
    /// indices, parameters, shape — the shared column format of
    /// [`Network::describe`](crate::Network::describe) and
    /// [`Plan::describe`](crate::Plan::describe), also this type's
    /// [`Display`](std::fmt::Display).
    pub(crate) fn spec_line(&self) -> String {
        let mut detail = String::new();
        let operands: Vec<String> = self
            .operands
            .iter()
            .map(|operand| operand.id.index().to_string())
            .collect();
        detail.push_str(&operands.join(", "));
        let parameters = self.opcode.parameter_text();
        if !parameters.is_empty() {
            if !detail.is_empty() {
                detail.push_str("  ");
            }
            detail.push_str(&parameters);
        }
        let mut line = String::new();
        let _ = write!(
            line,
            "{:4}  {:<14} {:<18} {}",
            self.symbol.id.index(),
            self.name(),
            detail,
            self.shape,
        );
        line
    }
}

/// A node displays as its spec line: the one format `describe` prints.
impl std::fmt::Display for Node {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.spec_line())
    }
}

#[cfg(test)]
#[path = "tests/opcode_tests.rs"]
mod tests;
