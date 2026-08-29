use std::cell::RefCell;

use super::backend::Backend;
use super::formula::{Formula, Precision};

/// One tallied dispatch outcome: how many tasks of one formula and
/// precision one server took inside a [`Backend::tallied`] scope.
///
/// The offered chain is the one stage of the stack a dump cannot
/// show — which backend served is decided per task at run time — so
/// the tally is that stage's data: every task that reached the
/// chain lands in exactly one row. `None` names the reference
/// paths: every member declined, or the posture admitted none.
/// Payloads whose element never hooks the seam do not reach the
/// chain and are not counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Service {
    /// The formula the tasks spoke.
    pub formula: Formula,
    /// The forwarding precision of the tasks.
    pub precision: Precision,
    /// The backend that served, or `None` when the reference path
    /// computed the task.
    pub backend: Option<Backend>,
    /// How many tasks landed this way inside the scope.
    pub count: u64,
}

thread_local! {
    /// The open tally of this thread, if a scope is active; `note`
    /// is a no-op otherwise, so unmeasured dispatch costs one
    /// thread-local read per task.
    static TALLY: RefCell<Option<Vec<Service>>> = const { RefCell::new(None) };
}

/// Records one dispatch outcome into this thread's open tally, if
/// any.
pub(super) fn note(formula: Formula, precision: Precision, backend: Option<Backend>) {
    TALLY.with(|cell| {
        let mut tally = cell.borrow_mut();
        let Some(rows) = tally.as_mut() else {
            return;
        };
        if let Some(row) = rows.iter_mut().find(|row| {
            row.formula == formula && row.precision == precision && row.backend == backend
        }) {
            row.count += 1;
            return;
        }
        rows.push(Service {
            formula,
            precision,
            backend,
            count: 1,
        });
    });
}

/// Runs `body` with a fresh tally installed for this thread and
/// answers the rows it collected, restoring any enclosing tally on
/// every exit including panics (the interrupted rows are discarded).
pub(super) fn tallied<Output>(body: impl FnOnce() -> Output) -> (Output, Vec<Service>) {
    struct Scope {
        previous: Option<Vec<Service>>,
    }
    impl Drop for Scope {
        fn drop(&mut self) {
            TALLY.with(|cell| *cell.borrow_mut() = self.previous.take());
        }
    }

    let scope = Scope {
        previous: TALLY.with(|cell| cell.borrow_mut().replace(Vec::new())),
    };
    let output = body();
    let rows = TALLY
        .with(|cell| cell.borrow_mut().take())
        .expect("the scope installed a tally");
    drop(scope);
    (output, rows)
}

#[cfg(test)]
#[path = "tests/service_tests.rs"]
mod tests;
