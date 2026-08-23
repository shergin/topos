use std::ops::Add;

use static_assertions::assert_impl_all;

use crate::{Element, Tensor};

use super::Field;

use super::{Network, Origin, SlotStore, Symbol, ValueId};

// Request-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Parameters<f64>: Send, Sync);

/// A payload per parameter slot of one network family: the live
/// weights, and every other table aligned to them.
///
/// Where the [`Network`](crate::Network) is the immutable spec, this is
/// the state: born from the record-site initials
/// ([`Network::parameters`](crate::Network::parameters)) or a
/// checkpoint, passed by reference into every run and plan, and stepped
/// as pure data — no run mutates it, and training mints no new network.
/// `Clone` is honest and O(parameters), which is the whole cost of a
/// what-if: one spec, any number of states.
///
/// Live weights are one instance of the type; an update direction,
/// an optimizer moment, or the recorded gradients of a compiled
/// training run are other instances over the same slots. They share
/// the type because they share the invariant — alignment to the
/// parameter slots — and the algebra (`map`, `zip`, `scale`, `+`)
/// that optimizer state is built from. Nothing hides in the graph;
/// state lives in the caller's structs. The node-aligned analogue is
/// [`Field`](crate::Field), the research and teaching grain;
/// [`Field::parameters`](crate::Field::parameters) projects it onto
/// these slots.
#[derive(Debug, Clone)]
pub struct Parameters<E> {
    origin: Origin,
    store: SlotStore<Tensor<E>>,
}

impl<E: Element> Parameters<E> {
    /// Wraps `store` as a parameter-aligned table of the `origin`
    /// family.
    pub(crate) fn new(origin: Origin, store: SlotStore<Tensor<E>>) -> Self {
        Self { origin, store }
    }

    /// Builds a table of the `origin` family from `(node, payload)`
    /// rows in slot order: the engine's constructor for recorded
    /// gradients.
    pub(crate) fn from_rows(
        origin: Origin,
        rows: impl IntoIterator<Item = (ValueId, Tensor<E>)>,
    ) -> Self {
        Self {
            origin,
            store: SlotStore::from_rows(rows),
        }
    }

    /// Returns the origin token of the network family this state
    /// steps.
    pub(crate) fn origin(&self) -> Origin {
        self.origin
    }

    /// Returns the number of parameter slots.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Returns `true` if it carries no parameters.
    pub fn is_empty(&self) -> bool {
        self.store.len() == 0
    }

    /// Returns the payloads in slot order, for the engine's node
    /// evaluation.
    pub(crate) fn payloads(&self) -> &[Tensor<E>] {
        self.store.payloads()
    }

    /// Returns the payload of the parameter named by `symbol`.
    ///
    /// # Panics
    /// Panics if `symbol` belongs to a different network or does not
    /// name a parameter these parameters carry.
    pub fn of(&self, symbol: Symbol) -> &Tensor<E> {
        assert!(
            symbol.origin == self.origin,
            "symbol belongs to a different network"
        );
        let Some(slot) = self.store.slot_of(symbol.id) else {
            panic!("symbol does not name a parameter these parameters carry");
        };
        &self.store.payloads()[slot.index()]
    }

    /// Returns a table with every entry passed through `transform`.
    pub fn map(&self, transform: impl Fn(&Tensor<E>) -> Tensor<E>) -> Self {
        Self {
            origin: self.origin,
            store: self
                .store
                .with_payloads(self.store.payloads().iter().map(transform).collect()),
        }
    }

    /// Combines two tables entry by entry with `combine`.
    ///
    /// # Panics
    /// Panics if the tables belong to different networks or cover
    /// different parameter slots.
    pub fn zip(&self, other: &Self, combine: impl Fn(&Tensor<E>, &Tensor<E>) -> Tensor<E>) -> Self {
        self.assert_compatible(other);
        Self {
            origin: self.origin,
            store: self.store.with_payloads(
                self.store
                    .payloads()
                    .iter()
                    .zip(other.store.payloads())
                    .map(|(left, right)| combine(left, right))
                    .collect(),
            ),
        }
    }

    /// The parameter slots of this table, filled from `field`: the
    /// projection from the node grain to the slot grain, shared by
    /// [`Field::parameters`](crate::Field::parameters).
    ///
    /// # Panics
    /// Panics if `field` belongs to a different network or does not
    /// cover every parameter slot.
    pub(super) fn filled_from(&self, field: &Field<E>) -> Self {
        assert!(
            field.origin() == self.origin,
            "field belongs to a different network"
        );
        if let Some(last) = self.store.last_node() {
            assert!(
                last.index() < field.len(),
                "field is stale: it does not cover every parameter"
            );
        }
        let payloads = self
            .store
            .iter()
            .map(|(node, _)| field.payloads()[node.index()].clone())
            .collect();
        Self {
            origin: self.origin,
            store: self.store.with_payloads(payloads),
        }
    }

    /// Panics if `other` cannot combine with `self`.
    fn assert_compatible(&self, other: &Self) {
        assert!(
            self.origin == other.origin,
            "parameter tables belong to different networks"
        );
        assert_eq!(
            self.store.len(),
            other.store.len(),
            "parameter tables cover different slots"
        );
    }

    /// Returns the state with every payload replaced by
    /// `rule(current, direction)`: the training-step transition.
    ///
    /// `direction` is any parameter-aligned table of this network
    /// family: the recorded gradients of a compiled training run
    /// ([`Run::recorded_gradients`](crate::Run::recorded_gradients)),
    /// an engine backward projected through
    /// [`Field::parameters`](crate::Field::parameters), or a derived
    /// direction such as a momentum velocity. The step is pure data —
    /// O(parameters) work and allocations, no new network, no lock —
    /// and slot order is preserved, so symbols keep naming their
    /// parameters.
    ///
    /// # Panics
    /// Panics if `direction` belongs to a different network or covers
    /// different parameter slots, or if `rule` returns a payload whose
    /// shape differs from the parameter's.
    pub fn step(
        &self,
        direction: &Parameters<E>,
        mut rule: impl FnMut(&Tensor<E>, &Tensor<E>) -> Tensor<E>,
    ) -> Self {
        self.step_each(direction, move |_, current, direction| {
            rule(current, direction)
        })
    }

    /// Returns the state stepped like [`Parameters::step`], with the
    /// parameter's [`Symbol`] passed to the rule: the identity-aware
    /// form, for per-parameter policy — an optimizer's selective
    /// weight decay, per-parameter clipping, or logging — decided from
    /// the parameter's symbol or the payload's own shape at the call
    /// site.
    ///
    /// The rule runs once per parameter, in slot order (the order the
    /// parameters were recorded); an `FnMut` rule may observe that
    /// order, and it is part of the method's contract.
    ///
    /// # Panics
    /// Panics as [`Parameters::step`] panics.
    pub fn step_each(
        &self,
        direction: &Parameters<E>,
        mut rule: impl FnMut(Symbol, &Tensor<E>, &Tensor<E>) -> Tensor<E>,
    ) -> Self {
        assert!(
            direction.origin == self.origin,
            "direction belongs to a different network"
        );
        assert_eq!(
            direction.store.len(),
            self.store.len(),
            "direction covers different parameter slots"
        );
        let mut payloads = Vec::with_capacity(self.store.len());
        for ((node, current), direction) in self.store.iter().zip(direction.store.payloads()) {
            let symbol = Symbol {
                origin: self.origin,
                id: node,
            };
            let next = rule(symbol, current, direction);
            assert_eq!(
                next.shape(),
                current.shape(),
                "step must preserve the parameter's shape"
            );
            payloads.push(next);
        }
        Self {
            origin: self.origin,
            store: self.store.with_payloads(payloads),
        }
    }

    /// Returns the state carried across an
    /// [`Network::into_tape`](crate::Network::into_tape) round trip:
    /// existing slots keep these payloads, slots recorded since take
    /// their record-site initials.
    ///
    /// # Panics
    /// Panics if `network` belongs to a different family or records
    /// fewer parameters than this state carries.
    pub fn carried(&self, network: &Network<E>) -> Self {
        assert!(
            network.origin() == self.origin,
            "parameters belong to a different network"
        );
        let fresh = network.parameters();
        assert!(
            self.len() <= fresh.len(),
            "parameters cover more slots than the network records"
        );
        let mut payloads: Vec<Tensor<E>> = Vec::with_capacity(fresh.len());
        payloads.extend(self.store.payloads().iter().cloned());
        payloads.extend(fresh.store.payloads()[self.len()..].iter().cloned());
        Self {
            origin: self.origin,
            store: fresh.store.with_payloads(payloads),
        }
    }

    /// Returns the state with the named parameters' payloads replaced:
    /// the installation route for checkpoints and foreign weights.
    ///
    /// Every other slot keeps its payload; replacing the same symbol
    /// twice keeps the last entry.
    ///
    /// # Panics
    /// Panics if a symbol belongs to a different network or does not
    /// name a parameter, or if a replacement's shape differs from the
    /// parameter's.
    pub fn with_payloads(
        &self,
        replacements: impl IntoIterator<Item = (Symbol, Tensor<E>)>,
    ) -> Self {
        let mut payloads = self.store.payloads().to_vec();
        for (symbol, payload) in replacements {
            assert!(
                symbol.origin == self.origin,
                "symbol belongs to a different network"
            );
            let Some(slot) = self.store.slot_of(symbol.id) else {
                panic!("symbol does not name a parameter these parameters carry");
            };
            assert_eq!(
                payload.shape(),
                payloads[slot.index()].shape(),
                "a replacement must preserve the parameter's shape"
            );
            payloads[slot.index()] = payload;
        }
        Self {
            origin: self.origin,
            store: self.store.with_payloads(payloads),
        }
    }
}

impl<E: Element> Parameters<E> {
    /// Returns a table with every entry multiplied by the single-value
    /// `factor`, spread to each entry's shape.
    ///
    /// It is the scalar arithmetic of optimizer state: bias-correction
    /// and decay factors multiply every parameter's entry regardless of
    /// its shape. For rank-0 entries the spread is the identity.
    ///
    /// # Panics
    /// Panics if `factor` holds more than one value.
    pub fn scale(&self, factor: &Tensor<E>) -> Self {
        self.map(|value| value.clone() * factor.broadcast_like(value))
    }
}

impl<E: Element> Add for &Parameters<E> {
    type Output = Parameters<E>;

    fn add(self, rhs: Self) -> Parameters<E> {
        self.zip(rhs, |left, right| left.clone() + right.clone())
    }
}

impl<E: Element> Add for Parameters<E> {
    type Output = Parameters<E>;

    fn add(self, rhs: Self) -> Parameters<E> {
        &self + &rhs
    }
}

#[cfg(test)]
#[path = "tests/parameters_tests.rs"]
mod tests;
