use crate::graph::{Network, Parameters, Symbol};
use crate::{Element, Numerics, Plan, Run, Tensor};

/// A function exported from a network: the declared reading — roots,
/// observes, memory posture, numerics — that every executor takes.
///
/// The entry names its roots (what must be computed — a loss, a
/// logits head, recorded gradient symbols; no root is special), the
/// extra interior values to observe (readable after a run, alongside
/// the roots), and whether run buffers support the engine reverse
/// scan ([`Entry::backward()`]). One network exports any number of
/// entries sharing its parameters — the twin pattern, said in the
/// type system. An entry never touches the graph: recorded gradients
/// enter as ordinary roots, produced by a visible
/// [`Tape::differentiate`](crate::Tape::differentiate) beforehand, so
/// an entry is cheap and re-runnable.
///
/// The common road binds an entry to its network as it is built:
/// [`Network::entry`](crate::Network::entry) answers a
/// [`BoundEntry`](crate::BoundEntry) whose
/// [`interpret`](crate::BoundEntry::interpret) is the oracle over the
/// declared closure and whose [`lower`](crate::BoundEntry::lower)
/// derives the compiled [`Plan`](crate::Plan). Detached entries are
/// for storage: roots and observes are [`Symbol`]s, valid across
/// reopens, and [`Network::compile`](crate::Network::compile) binds
/// one late.
///
/// The four fields are the declaration itself, public to read: a
/// display or an external tool walks them directly. The builder
/// methods remain the construction road — [`Entry::roots()`] to
/// open, then chained `observe`, `backward`, and `numerics` —
/// converting through `Into<Symbol>` as they go.
///
/// # Examples
/// ```
/// # use topos::Tape;
/// # let tape = Tape::new();
/// # let weight = tape.parameter(1.0_f64);
/// # let loss = (weight * weight).sum().symbol();
/// # let network = tape.into_network();
/// // Pure inference: a forward-only plan over one root.
/// let inference = network.entry([loss]).lower();
///
/// // Engine training: run buffers retain what `backward` reads.
/// let training = network.entry([loss]).backward().lower();
/// assert!(!inference.can_backward());
/// assert!(training.can_backward());
/// ```
#[derive(Debug, Clone)]
pub struct Entry {
    /// The closure sources a run must compute; every root is
    /// readable after a run.
    pub roots: Vec<Symbol>,
    /// The interior values the caller also declared readable,
    /// alongside the roots.
    pub observe: Vec<Symbol>,
    /// Whether run buffers retain what the engine reverse scan
    /// reads.
    pub backward: bool,
    /// The numerics posture the entry's runs execute under.
    pub numerics: Numerics,
}

impl Entry {
    /// Opens a request over `roots`, the closure sources a run must
    /// compute; every root is readable after a run.
    pub fn roots(roots: impl IntoIterator<Item = impl Into<Symbol>>) -> Self {
        Self {
            roots: roots.into_iter().map(Into::into).collect(),
            observe: Vec::new(),
            backward: false,
            numerics: Numerics::Fast,
        }
    }

    /// Adds interior values the caller also wants readable after a
    /// run; like roots, they seed the plan's reachability closure.
    /// Repeated calls accumulate.
    pub fn observe(mut self, extra: impl IntoIterator<Item = impl Into<Symbol>>) -> Self {
        self.observe.extend(extra.into_iter().map(Into::into));
        self
    }

    /// Entrys runs that answer [`Run::backward`](crate::Run::backward):
    /// buffers retain what the engine's reverse scan reads — the
    /// retain-all posture, which the graded consumers preferred on
    /// both axes over freeing or rematerializing mid-run. A request
    /// that never calls this compiles a forward-only plan, whose runs
    /// refuse `backward`; [`Plan::can_backward`](crate::Plan::can_backward)
    /// answers which kind a plan is.
    ///
    /// This is a memory posture, not a second compiler. For compiled
    /// training, prefer recording the derivative with
    /// [`Tape::differentiate`](crate::Tape::differentiate) and
    /// compiling a forward-only plan over the adjoints' roots: fusion
    /// and liveness then apply to the chain rule itself. `backward`'s
    /// place is the oracle — an engine reverse scan over a plan that
    /// did not record its derivative, for verification and quick
    /// procedural use.
    pub fn backward(mut self) -> Self {
        self.backward = true;
        self
    }

    /// Chooses the numerics posture of the plan's runs.
    /// [`Numerics::Fast`] — the default — engages the compiled backend
    /// chain above its cost thresholds; [`Numerics::Exact`] makes the
    /// chain decline every task, so runs compute on the built-in
    /// reference paths, bit-identical to the default build in every
    /// build. Reordering float math is always this labeled choice,
    /// never a silent effect of a feature flag — and an `Exact` and a
    /// `Fast` plan over the same network make the two results
    /// comparable in one process.
    pub fn numerics(mut self, numerics: Numerics) -> Self {
        self.numerics = numerics;
        self
    }
}

/// An [`Entry`] bound to the network that will execute it: the
/// builder the common road goes through, and the type that carries
/// the two executor verbs.
///
/// `interpret` is the oracle over the entry's closure — the
/// interpreter, evaluating exactly the ancestors of the declared
/// results. `lower` derives the compiled [`Plan`]: same closure,
/// plus keep-set, liveness, and pattern election. The bound entry
/// borrows its network, so an entry of a consumed network is
/// unrepresentable; [`into_entry`](BoundEntry::into_entry) detaches
/// the signature for storage across reopens.
#[derive(Debug)]
pub struct BoundEntry<'network, E> {
    network: &'network Network<E>,
    pub(crate) entry: Entry,
}

impl<'network, E: Element> BoundEntry<'network, E> {
    /// Adds interior values the caller also wants readable, exactly
    /// as [`Entry::observe()`].
    pub fn observe(mut self, extra: impl IntoIterator<Item = impl Into<Symbol>>) -> Self {
        self.entry = self.entry.observe(extra);
        self
    }

    /// Requests runs that answer engine `backward`, exactly as
    /// [`Entry::backward()`].
    pub fn backward(mut self) -> Self {
        self.entry = self.entry.backward();
        self
    }

    /// Chooses the numerics posture, exactly as [`Entry::numerics()`].
    pub fn numerics(mut self, numerics: Numerics) -> Self {
        self.entry = self.entry.numerics(numerics);
        self
    }

    /// Interprets the entry: the oracle, evaluating the declared
    /// results' ancestor closure with `feeds` bound to declared
    /// inputs for this run only. Values outside the closure hold
    /// placeholders, and reads of them panic — observability is
    /// declared, never inferred.
    ///
    /// # Panics
    /// Panics as [`Network::forward`](crate::Network::forward) panics,
    /// or if a declared symbol does not resolve in the network.
    pub fn interpret(
        &self,
        parameters: &Parameters<E>,
        feeds: impl IntoIterator<Item = (Symbol, Tensor<E>)>,
    ) -> Run<E> {
        self.network.interpret_entry(&self.entry, parameters, feeds)
    }

    /// Lowers the entry into a compiled [`Plan`]: the same declared
    /// reading, derived into a schedule with keep-set, liveness, and
    /// pattern election.
    ///
    /// # Panics
    /// Panics if a declared symbol does not resolve in the network.
    pub fn lower(&self) -> Plan<E> {
        self.network.compile(self.entry.clone())
    }

    /// Returns the declared reading this bound entry carries: the
    /// borrowing twin of [`into_entry`](BoundEntry::into_entry).
    pub fn entry(&self) -> &Entry {
        &self.entry
    }

    /// Detaches the entry for storage: symbols stay valid across
    /// reopens, and [`Network::compile`](crate::Network::compile)
    /// binds it again later.
    pub fn into_entry(self) -> Entry {
        self.entry
    }
}

impl<E: Element> Network<E> {
    /// Opens an entry over `roots` bound to this network: the common
    /// road to [`interpret`](BoundEntry::interpret) and
    /// [`lower`](BoundEntry::lower).
    pub fn entry(&self, roots: impl IntoIterator<Item = impl Into<Symbol>>) -> BoundEntry<'_, E> {
        BoundEntry {
            network: self,
            entry: Entry::roots(roots),
        }
    }
}

#[cfg(test)]
#[path = "tests/entry_tests.rs"]
mod tests;
