use super::{Adjoints, Symbol, Value};
use crate::Element;

/// The one-call mass detach of a keep-set: recording-phase handles
/// become the names later phases speak.
///
/// `[w, x, y, loss].keep()` turns an array, `Vec`, or mixed tuple of
/// [`Value`]s, [`Symbol`]s, and [`Adjoints`] into its detached form
/// in one call — the construction keep-set as a value instead of an
/// N-way `.symbol()` destructure. [`Tape::record`](super::Tape::record)
/// bounds its closure's return on this trait, whose non-`Value`
/// impls are identities, so only names leave the recording phase;
/// a returned `Value` is rejected by the borrow checker before the
/// bound is ever consulted.
///
/// Like `Into`, this is a conversion trait, not an extension point:
/// it is not object safe and does not need to be. A user struct of
/// `Symbol`s needs no impl — build it from `.symbol()` calls inside
/// the closure and it is already detached.
pub trait Keep {
    /// The detached form of this keep-set.
    type Kept;

    /// Detaches the names.
    fn keep(self) -> Self::Kept;
}

impl<E: Element> Keep for Value<'_, E> {
    type Kept = Symbol;

    fn keep(self) -> Symbol {
        self.symbol()
    }
}

/// Identity: already a detached name.
impl Keep for Symbol {
    type Kept = Symbol;

    fn keep(self) -> Symbol {
        self
    }
}

/// Identity: a differentiation product is already detached names.
impl Keep for Adjoints {
    type Kept = Adjoints;

    fn keep(self) -> Adjoints {
        self
    }
}

/// The empty keep-set, for recordings read back entirely by later
/// resolution.
impl Keep for () {
    type Kept = ();

    fn keep(self) {}
}

impl<T: Keep, const N: usize> Keep for [T; N] {
    type Kept = [T::Kept; N];

    fn keep(self) -> Self::Kept {
        self.map(Keep::keep)
    }
}

impl<T: Keep> Keep for Vec<T> {
    type Kept = Vec<T::Kept>;

    fn keep(self) -> Self::Kept {
        self.into_iter().map(Keep::keep).collect()
    }
}

/// Generates the tuple impls: mixed keep-sets such as
/// `(parameters, inputs, loss)` detach member by member.
macro_rules! keep_tuples {
    ($(($($name:ident),+))+) => {$(
        impl<$($name: Keep),+> Keep for ($($name,)+) {
            type Kept = ($($name::Kept,)+);

            #[allow(non_snake_case)]
            fn keep(self) -> Self::Kept {
                let ($($name,)+) = self;
                ($($name.keep(),)+)
            }
        }
    )+};
}

keep_tuples! {
    (A)
    (A, B)
    (A, B, C)
    (A, B, C, D)
    (A, B, C, D, E2)
    (A, B, C, D, E2, F)
}

#[cfg(test)]
#[path = "tests/keep_tests.rs"]
mod tests;
