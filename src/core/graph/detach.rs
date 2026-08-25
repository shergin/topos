use super::{Adjoints, Symbol, Value};
use crate::Element;

/// The one-call mass detach: recording-phase handles become the
/// names later phases speak.
///
/// `[w, x, y, loss].detach()` turns an array, `Vec`, or mixed tuple
/// of [`Value`]s, [`Symbol`]s, and [`Adjoints`] into its detached
/// form in one call, instead of an N-way `.symbol()` destructure.
/// [`Tape::record`](super::Tape::record) bounds its closure's return
/// on this trait, whose non-`Value` impls are identities, so only
/// names leave the recording phase; a returned `Value` is rejected by
/// the borrow checker before the bound is ever consulted.
///
/// What construction detaches is not an execution keep-set, and the
/// two words stay apart on purpose: these are the names later phases
/// may mention, while an [`Entry`](crate::Entry) declares what one
/// run computes and what it may read.
///
/// Like `Into`, this is a conversion trait, not an extension point:
/// it is not object safe and does not need to be. A user struct of
/// `Symbol`s needs no impl — build it from `.symbol()` calls inside
/// the closure and it is already detached.
pub trait Detach {
    /// The detached form of these names.
    type Detached;

    /// Detaches the names.
    fn detach(self) -> Self::Detached;
}

impl<E: Element> Detach for Value<'_, E> {
    type Detached = Symbol;

    fn detach(self) -> Symbol {
        self.symbol()
    }
}

/// Identity: already a detached name.
impl Detach for Symbol {
    type Detached = Symbol;

    fn detach(self) -> Symbol {
        self
    }
}

/// Identity: a differentiation product is already detached names.
impl Detach for Adjoints {
    type Detached = Adjoints;

    fn detach(self) -> Adjoints {
        self
    }
}

/// Nothing detached, for recordings read back entirely by later
/// resolution.
impl Detach for () {
    type Detached = ();

    fn detach(self) {}
}

impl<T: Detach, const N: usize> Detach for [T; N] {
    type Detached = [T::Detached; N];

    fn detach(self) -> Self::Detached {
        self.map(Detach::detach)
    }
}

impl<T: Detach> Detach for Vec<T> {
    type Detached = Vec<T::Detached>;

    fn detach(self) -> Self::Detached {
        self.into_iter().map(Detach::detach).collect()
    }
}

/// Generates the tuple impls: mixed groups such as
/// `(parameters, inputs, loss)` detach member by member.
macro_rules! detach_tuples {
    ($(($($name:ident),+))+) => {$(
        impl<$($name: Detach),+> Detach for ($($name,)+) {
            type Detached = ($($name::Detached,)+);

            #[allow(non_snake_case)]
            fn detach(self) -> Self::Detached {
                let ($($name,)+) = self;
                ($($name.detach(),)+)
            }
        }
    )+};
}

detach_tuples! {
    (A)
    (A, B)
    (A, B, C)
    (A, B, C, D)
    (A, B, C, D, E2)
    (A, B, C, D, E2, F)
}

#[cfg(test)]
#[path = "tests/detach_tests.rs"]
mod tests;
