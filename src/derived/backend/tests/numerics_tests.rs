//! Pins the exact posture's bits across builds and platforms.
//!
//! One graph whose product and map exceed the host backends' cost
//! thresholds runs on the two exact roads — `Network::forward`,
//! exact by construction, and an `Exact`-postured plan — and the
//! result bits, folded to one digest per result, are asserted
//! against pinned constants. Every feature build and every platform
//! checks the same numbers, so this is the cross-build spelling of
//! "the interpreter's bits are the truth": a backend serving under
//! `Exact`, or a reference kernel drifting between targets, fails
//! here first.

use crate::{Element, Numerics, Tape, Tensor};

/// FNV-1a over every element's bit pattern, in iteration order.
fn digest<E: Element>(tensor: &Tensor<E>, bits: impl Fn(E) -> u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for element in tensor.iter() {
        hash ^= bits(element);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

/// A deterministic dense payload no backend can shortcut: every
/// element distinct, derived from an integer formula.
fn dense(index: usize, seed: u64) -> f64 {
    let mixed = (index as u64)
        .wrapping_mul(2_654_435_761)
        .wrapping_add(seed)
        % 2000;
    (mixed as f64 - 1000.0) / 500.0
}

/// The digests of `product`, `activated`, and `loss`, in that order,
/// through `Network::forward` and through an `Exact` plan — both
/// must answer the pinned reference bits.
fn exact_digests<E: Element>(
    make: impl Fn(f64) -> E + Copy,
    bits: impl Fn(E) -> u64 + Copy,
) -> [u64; 3] {
    let tape: Tape<E> = Tape::new();
    let inputs = tape.leaf(Tensor::new(
        [64, 128],
        (0..64 * 128)
            .map(|index| make(dense(index, 1)))
            .collect::<Vec<_>>(),
    ));
    let weights = tape.parameter(Tensor::new(
        [128, 96],
        (0..128 * 96)
            .map(|index| make(dense(index, 2)))
            .collect::<Vec<_>>(),
    ));
    // 2 * 64 * 128 * 96 = 1.57M FLOPs and 6144 map elements: above
    // the host backends' gemm and map thresholds, so under `Fast`
    // these would be served and the digests would depend on the
    // build. Under the exact roads they must not be.
    let product = inputs.matmul(weights);
    let activated = product.tanh();
    let scores = activated.log_softmax(1);
    let loss = (scores * scores).sum();
    let [product, activated, loss] = [product.symbol(), activated.symbol(), loss.symbol()];
    let network = tape.into_network();
    let parameters = network.parameters();

    let forward = network.forward(&parameters, []);
    let plan = network
        .entry([loss])
        .observe([product, activated])
        .numerics(Numerics::Exact)
        .lower();
    let planned = plan.forward(&parameters, std::iter::empty());

    let symbols = [product, activated, loss];
    let from_forward = symbols.map(|symbol| digest(forward.of(symbol), bits));
    let from_plan = symbols.map(|symbol| digest(planned.of(symbol), bits));
    assert_eq!(
        from_forward, from_plan,
        "the exact plan must answer the whole-spec oracle's bits"
    );
    from_forward
}

#[test]
fn exact_f64_bits_are_identical_in_every_build() {
    // Recorded from the default build on 2026-08-29; every feature
    // build and platform must reproduce them.
    const PINNED: [u64; 3] = [
        0xC993_A366_A773_52C3,
        0xD38A_AD8A_68A7_7E68,
        0x2A18_1A78_1782_88B9,
    ];
    assert_eq!(
        exact_digests::<f64>(|value| value, f64::to_bits),
        PINNED,
        "exact f64 bits drifted from the pinned reference"
    );
}

#[test]
fn exact_f32_bits_are_identical_in_every_build() {
    // Recorded from the default build on 2026-08-29; every feature
    // build and platform must reproduce them.
    const PINNED: [u64; 3] = [
        0xF347_AE30_D86A_C54D,
        0x56F7_2C65_4764_59B5,
        0xC8BD_BCC7_0915_A42C,
    ];
    assert_eq!(
        exact_digests::<f32>(|value| value as f32, |element| u64::from(element.to_bits())),
        PINNED,
        "exact f32 bits drifted from the pinned reference"
    );
}
