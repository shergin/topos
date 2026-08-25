//! The bitwise reference kernels, published for differential testing.
//!
//! Every fast path in the crate is graded against a composed reference
//! that computes the same bits through the plain payload operations.
//! This module re-exports those references so an out-of-tree element
//! type can be tested exactly the way the in-crate ones are: run the
//! fast path, run the reference, assert bit equality.
//!
//! [`multiply`] is the slice-path matrix multiplication every
//! [`Elementary::gemm`](crate::Elementary::gemm) hook must reproduce.
//! The three `composed_*` formulas are the recorded-order references
//! behind the fused executors on [`Tensor`](crate::Tensor).

pub use crate::payload::gemm::multiply;
pub use crate::payload::recordable::{
    composed_batch_norm, composed_max_pool, composed_windowed_patches,
};
