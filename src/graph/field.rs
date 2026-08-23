use std::ops::Add;

use static_assertions::assert_impl_all;

use crate::{Differentiable, Tensorial};

use super::{Origin, Parameters, Symbol};

// Request-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Field<f64>: Send, Sync);

/// A value-aligned buffer over the nodes of one network's recording.
///
/// The [`Gradients`] of a backward run are one kind of field, and a
/// [`Run`](crate::Run) holds its forward payloads in one too. The node
/// grain is the research and teaching product — `gradients.of(hidden)`
/// answers for every value, not only parameters — while training
/// speaks the parameter grain: [`Field::parameters`] projects a field
/// onto a [`Parameters`](crate::Parameters) table, whose slot-aligned
/// algebra carries optimizer state. Fields carry their network
/// family's origin rather than borrowing anything, so a field
/// outlives every phase.
///
/// Field operations require both operands to cover the same number of nodes
/// of the same network. A field produced before a reopen extends the
/// recording still covers its original prefix; accessing a newer node or
/// projecting onto parameters it does not cover is rejected.
#[derive(Debug, Clone)]
pub struct Field<Data> {
    origin: Origin,
    payloads: Vec<Data>,
}

/// The gradients of one backward run: the derivative of the run's target with
/// respect to every node.
///
/// It is an alias rather than a distinct type because gradients *are* a field,
/// the one that differentiation produces, so every field operation applies to
/// them unchanged. Read a single gradient with [`Field::of`], and project the
/// parameter entries out for training with [`Field::parameters`]. The alias
/// names the role at the API boundary, most visibly on
/// [`Run::backward`](crate::Run::backward), while the type keeps
/// the one invariant it actually enforces: alignment to a graph, not
/// differentiation.
pub type Gradients<Data> = Field<Data>;

impl<Data: Differentiable> Field<Data> {
    pub(crate) fn new(origin: Origin, payloads: Vec<Data>) -> Self {
        Self { origin, payloads }
    }

    /// Returns the origin token of the network family this field
    /// covers.
    pub(crate) fn origin(&self) -> Origin {
        self.origin
    }

    /// Returns the number of nodes this field covers.
    pub(crate) fn len(&self) -> usize {
        self.payloads.len()
    }

    /// Returns the value assigned to the node named by `symbol`.
    ///
    /// # Panics
    /// Panics if `symbol` belongs to a different network or was
    /// allocated after this field was produced.
    pub fn of(&self, symbol: Symbol) -> &Data {
        assert!(
            symbol.origin == self.origin,
            "symbol belongs to a different network"
        );
        assert!(
            symbol.id.index() < self.payloads.len(),
            "symbol was allocated after this field was produced"
        );
        &self.payloads[symbol.id.index()]
    }

    /// Returns a field with every entry passed through `transform`.
    pub fn map(&self, transform: impl Fn(&Data) -> Data) -> Self {
        Self {
            origin: self.origin,
            payloads: self.payloads.iter().map(transform).collect(),
        }
    }

    /// Combines two fields entry by entry with `combine`.
    ///
    /// # Panics
    /// Panics if the fields belong to different networks or cover
    /// different numbers of nodes.
    pub fn zip(&self, other: &Self, combine: impl Fn(&Data, &Data) -> Data) -> Self {
        self.assert_compatible(other);
        Self {
            origin: self.origin,
            payloads: self
                .payloads
                .iter()
                .zip(&other.payloads)
                .map(|(left, right)| combine(left, right))
                .collect(),
        }
    }

    /// Returns every node's payload in tape order, for engine scans
    /// and the displays that plot a whole field rather than read one
    /// value out of it.
    pub(crate) fn payloads(&self) -> &[Data] {
        &self.payloads
    }

    /// Returns the parameter slots of `parameters`, filled from this
    /// field: the projection from the node grain to the slot grain.
    ///
    /// A complete field is the research and teaching product — every
    /// cotangent readable — while training speaks parameter alignment;
    /// this is the bridge, so an engine
    /// [`backward`](crate::Run::backward) feeds
    /// [`Parameters::step`](crate::Parameters::step) as
    /// `run.backward(loss).parameters(&parameters)`.
    ///
    /// # Panics
    /// Panics if `parameters` belongs to a different network or this
    /// field does not cover every parameter slot (it is stale: the
    /// recording grew parameters after the field was produced).
    pub fn parameters(&self, parameters: &Parameters<Data>) -> Parameters<Data> {
        parameters.filled_from(self)
    }

    /// Panics if `other` cannot combine with `self`.
    fn assert_compatible(&self, other: &Self) {
        assert!(
            self.origin == other.origin,
            "fields belong to different networks"
        );
        assert_eq!(
            self.payloads.len(),
            other.payloads.len(),
            "fields cover different prefixes of the network"
        );
    }
}

impl<Data: Tensorial> Field<Data> {
    /// Returns a field with every entry multiplied by the single-value
    /// `factor`, spread to each entry's shape.
    ///
    /// It is the scalar arithmetic of whole-graph analysis — weighting
    /// a run's cotangents before combining them with another run's.
    /// For scalar payloads the spread is the identity, so scalar
    /// fields scale exactly as they always did.
    ///
    /// # Panics
    /// For tensor payloads, panics if `factor` holds more than one
    /// value.
    pub fn scale(&self, factor: &Data) -> Self {
        self.map(|value| value.clone() * factor.broadcast_like(value))
    }
}

impl<Data: Differentiable> Add for &Field<Data> {
    type Output = Field<Data>;

    fn add(self, rhs: Self) -> Field<Data> {
        self.zip(rhs, |left, right| left.clone() + right.clone())
    }
}

impl<Data: Differentiable> Add for Field<Data> {
    type Output = Field<Data>;

    fn add(self, rhs: Self) -> Field<Data> {
        &self + &rhs
    }
}

#[cfg(test)]
#[path = "tests/field_tests.rs"]
mod tests;
