mod bf16;
mod differentiable;
mod element;
mod elementary;
// Reachable for the `reference` facade's re-exports.
pub(crate) mod gemm;
mod layout;
mod normalized;
mod shape;
mod storage;
mod tensor;
// Reachable for the `reference` facade's re-exports.
pub(crate) mod tensorial;

pub use bf16::Bf16;
pub use differentiable::Differentiable;
pub use element::Element;
pub use elementary::{Elementary, MapOperation};
pub use gemm::GemmTask;
pub use normalized::{BatchNormTask, Normalized};
pub use shape::Shape;
pub use tensor::Tensor;
pub use tensorial::Tensorial;
