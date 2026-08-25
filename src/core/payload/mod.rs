mod bf16;
mod differentiable;
mod element;
mod elementary;
mod erf;
// Reachable for the `reference` facade's re-exports.
pub(crate) mod gemm;
mod layout;
mod normalized;
// Reachable for the `reference` facade's re-exports.
pub(crate) mod recordable;
mod shape;
mod storage;
mod tensor;

pub use bf16::Bf16;
pub use differentiable::Differentiable;
pub use element::Element;
pub use elementary::{Elementary, MapOperation};
pub use gemm::GemmTask;
pub use normalized::{BatchNormTask, Normalized};
pub use recordable::Recordable;
pub use shape::Shape;
pub use tensor::Tensor;
