use super::SlotId;

/// A learnable parameter: a leaf whose live payload a training step
/// replaces through `Parameters::step`.
///
/// The node holds only its slot; the recorded initial lives in the
/// network's slot table and the live payload in the caller's
/// `Parameters`, which is what lets a gradient step swap state without
/// touching the recorded structure. It behaves exactly like `Leaf`
/// during runs: supplied rather than computed, with no gradients
/// routed back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Parameter(pub(crate) SlotId);
