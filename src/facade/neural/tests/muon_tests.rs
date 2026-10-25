use crate::{AdamW, Muon, NewtonSchulz, Optimizer, Parameters, Tape, Tensor};

/// The reference implementation's tuned quintic over five iterations
/// with its normalization stabilizer, as single-value payloads.
fn quintic() -> NewtonSchulz<f64> {
    NewtonSchulz::new(
        [
            Tensor::from(3.4445_f64),
            Tensor::from(-4.7750_f64),
            Tensor::from(2.0315_f64),
        ],
        5,
        Tensor::from(1e-7_f64),
    )
}

/// The conventional Muon: momentum `0.95` over the reference quintic.
fn conventional() -> Muon<f64> {
    Muon::new(Tensor::from(0.95_f64), quintic())
}

/// The scalar twin of the quintic iteration, the map every singular
/// value of a normalized matrix passes through, in the operation order
/// the tensor form uses.
fn iterate_scalar(value: f64, steps: usize) -> f64 {
    let (a, b, c) = (3.4445_f64, -4.7750_f64, 2.0315_f64);
    let mut x = value;
    for _ in 0..steps {
        let gram = x * x;
        let blend = gram * b + (gram * gram) * c;
        x = x * a + blend * x;
    }
    x
}

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < tolerance,
            "{actual} is not within {tolerance} of {expected}"
        );
    }
}

fn assert_bitwise(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{actual} != {expected}"
        );
    }
}

#[test]
fn newton_schulz_maps_each_singular_value_through_the_polynomial() {
    // A diagonal matrix stays diagonal under the iteration, so its
    // entries are its singular values (with sign) read directly, each
    // mapped through the scalar twin of the polynomial.
    let diagonal = Tensor::new([3, 3], [3.0_f64, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0]);
    let orthogonal = quintic().orthogonalize(&diagonal).to_vec();
    let norm = (9.0_f64 + 0.25 + 1.0).sqrt() + 1e-7;
    for (index, entry) in [3.0_f64, -0.5, 1.0].into_iter().enumerate() {
        let expected = iterate_scalar(entry / norm, 5);
        let actual = orthogonal[index * 4];
        assert!((actual - expected).abs() < 1e-12, "{actual} vs {expected}");
        // The quintic lands near one, keeping the sign.
        assert!(
            (0.6..=1.2).contains(&actual.abs()),
            "{actual} is off the band"
        );
        assert_eq!(actual.is_sign_negative(), entry.is_sign_negative());
    }
    for (index, value) in orthogonal.iter().enumerate() {
        if index % 4 != 0 {
            assert_eq!(*value, 0.0);
        }
    }
}

#[test]
fn cubic_newton_schulz_converges_to_the_polar_factor() {
    // The classical cubic has one as its exact fixed point, so enough
    // iterations recover the polar factor to machine precision: for a
    // scaled permutation, the permutation itself.
    let cubic = NewtonSchulz::new(
        [
            Tensor::from(1.5_f64),
            Tensor::from(-0.5_f64),
            Tensor::from(0.0_f64),
        ],
        24,
        Tensor::from(0.0_f64),
    );
    let scaled_permutation = Tensor::new([2, 2], [0.0_f64, 2.0, 1.0, 0.0]);
    let polar = cubic.orthogonalize(&scaled_permutation);
    assert_close(&polar.to_vec(), &[0.0, 1.0, 1.0, 0.0], 1e-12);

    // A tall matrix comes back with orthonormal columns.
    let tall = Tensor::new([3, 2], [2.0_f64, 1.0, 0.0, 1.0, 1.0, -1.0]);
    let orthogonal = cubic.orthogonalize(&tall);
    assert_eq!(orthogonal.shape().axes(), [3, 2]);
    let gram = orthogonal.transpose().matmul(&orthogonal);
    assert_close(&gram.to_vec(), &[1.0, 0.0, 0.0, 1.0], 1e-12);
}

#[test]
fn newton_schulz_is_transpose_equivariant_and_scale_invariant() {
    let matrix = Tensor::new([3, 2], [2.0_f64, 1.0, 0.0, 1.0, 1.0, -1.0]);
    let orthogonal = quintic().orthogonalize(&matrix);
    let transposed = quintic().orthogonalize(&matrix.transpose());
    assert_close(
        &orthogonal.transpose().to_vec(),
        &transposed.to_vec(),
        1e-12,
    );

    // Without the stabilizer the normalization cancels any scale.
    let unstabilized = NewtonSchulz::new(
        [
            Tensor::from(3.4445_f64),
            Tensor::from(-4.7750_f64),
            Tensor::from(2.0315_f64),
        ],
        5,
        Tensor::from(0.0_f64),
    );
    let scaled = matrix.clone() * Tensor::from(7.0_f64).broadcast_like(&matrix);
    assert_close(
        &unstabilized.orthogonalize(&scaled).to_vec(),
        &unstabilized.orthogonalize(&matrix).to_vec(),
        1e-12,
    );
}

#[test]
fn two_steps_match_a_hand_trace_on_a_diagonal_weight() {
    // `loss = sum(w * w)` over a diagonal `[2, 2]` weight: the gradient
    // `2w` is diagonal, so the orthogonalized direction is the scalar
    // polynomial applied to each normalized diagonal entry, and the
    // whole trajectory is traced by hand.
    let tape = Tape::new();
    let w = tape.parameter(Tensor::new([2, 2], [1.0_f64, 0.0, 0.0, 2.0]));
    let loss = (w * w).sum().symbol();
    let w = w.symbol();
    let network = tape.into_network();
    let mut muon = conventional();
    let rate = Tensor::from(0.1_f64);

    let mut parameters = network.parameters();
    let mut expected = [1.0_f64, 2.0];
    let mut buffer = [0.0_f64, 0.0];
    for _ in 0..2 {
        let run = network.forward(&parameters, []);
        let gradients = run.backward(loss).parameters(&parameters);
        parameters = muon.step(&parameters, &gradients, &rate);

        let gradient = expected.map(|entry| 2.0 * entry);
        for axis in 0..2 {
            buffer[axis] = buffer[axis] * 0.95 + gradient[axis];
        }
        let direction = [
            gradient[0] + buffer[0] * 0.95,
            gradient[1] + buffer[1] * 0.95,
        ];
        let norm = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt() + 1e-7;
        for axis in 0..2 {
            expected[axis] -= 0.1 * iterate_scalar(direction[axis] / norm, 5);
        }

        let stepped = parameters.of(w).to_vec();
        assert_close(&[stepped[0], stepped[3]], &expected, 1e-12);
        assert_eq!(stepped[1], 0.0);
        assert_eq!(stepped[2], 0.0);
    }
}

#[test]
fn identical_runs_are_bitwise_identical() {
    let run = || {
        let tape = Tape::new();
        let w = tape.parameter(Tensor::new([2, 2], [1.0_f64, -0.5, 0.25, 2.0]));
        let x = tape.leaf(Tensor::new([3, 2], [1.0_f64, -1.0, 0.5, 2.0, -1.5, 0.25]));
        let product = x.matmul(w);
        let loss = (product * product).sum().symbol();
        let w = w.symbol();
        let network = tape.into_network();
        let mut muon = conventional();
        let rate = Tensor::from(0.05_f64);
        let mut parameters = network.parameters();
        for _ in 0..5 {
            let run = network.forward(&parameters, []);
            let gradients = run.backward(loss).parameters(&parameters);
            parameters = muon.step(&parameters, &gradients, &rate);
        }
        parameters.of(w).to_vec()
    };
    assert_bitwise(&run(), &run());
}

#[test]
fn matrices_are_orthogonalized_and_vectors_take_the_momentum_direction() {
    let tape = Tape::new();
    let weights = tape.parameter(Tensor::new([2, 2], [1.0_f64, -0.5, 0.25, 2.0]));
    let bias = tape.parameter(Tensor::new([2], [0.5_f64, -1.0]));
    let x = tape.leaf(Tensor::new([1, 2], [1.0_f64, -1.0]));
    let product = x.matmul(weights);
    let shifted = product + bias.broadcast_along_like(0, product);
    let loss = (shifted * shifted).sum();
    let (weights, bias, loss) = (weights.symbol(), bias.symbol(), loss.symbol());
    let network = tape.into_network();
    let mut muon = conventional();
    let rate = Tensor::from(0.05_f64);

    let parameters = network.parameters();
    let run = network.forward(&parameters, []);
    let gradients = run.backward(loss).parameters(&parameters);
    let stepped = muon.step(&parameters, &gradients, &rate);

    // On the first step the buffer is the gradient, so the Nesterov
    // direction is `g + 0.95 g` in exactly that arithmetic.
    let nesterov = |gradient: &Tensor<f64>| {
        gradient.clone() + gradient.clone() * Tensor::from(0.95_f64).broadcast_like(gradient)
    };
    let direction = nesterov(gradients.of(bias));
    let expected_bias =
        parameters.of(bias).clone() - direction.clone() * rate.broadcast_like(&direction);
    assert_bitwise(&stepped.of(bias).to_vec(), &expected_bias.to_vec());

    // The square weight orthogonalizes at scale one.
    let orthogonal = quintic().orthogonalize(&nesterov(gradients.of(weights)));
    let expected_weights =
        parameters.of(weights).clone() - orthogonal.clone() * rate.broadcast_like(&orthogonal);
    assert_bitwise(&stepped.of(weights).to_vec(), &expected_weights.to_vec());
}

#[test]
fn step_where_overrides_the_structural_policy() {
    let tape = Tape::new();
    let weights = tape.parameter(Tensor::new([2, 2], [1.0_f64, -0.5, 0.25, 2.0]));
    let loss = (weights * weights).sum().symbol();
    let weights = weights.symbol();
    let network = tape.into_network();
    let mut muon = conventional();
    let rate = Tensor::from(0.05_f64);

    let parameters = network.parameters();
    let run = network.forward(&parameters, []);
    let gradients = run.backward(loss).parameters(&parameters);
    // Orthogonalize nothing: the matrix steps like a vector.
    let stepped = muon.step_where(&parameters, &gradients, &rate, |_, _| false);

    let gradient = gradients.of(weights);
    let direction =
        gradient.clone() + gradient.clone() * Tensor::from(0.95_f64).broadcast_like(gradient);
    let expected =
        parameters.of(weights).clone() - direction.clone() * rate.broadcast_like(&direction);
    assert_bitwise(&stepped.of(weights).to_vec(), &expected.to_vec());
}

#[test]
fn tall_matrices_are_scaled_by_their_aspect_ratio() {
    // `loss = sum(w * c)` makes `c` the gradient outright; the tall
    // weight's gradient is the wide weight's transposed, so their
    // orthogonalized updates are transposes of each other up to the
    // tall one's `sqrt(4 / 2)` scale.
    let tape = Tape::new();
    let tall = tape.parameter(Tensor::filled([4, 2], 0.0_f64));
    let wide = tape.parameter(Tensor::filled([2, 4], 0.0_f64));
    let slope = Tensor::new([4, 2], [2.0_f64, 1.0, 0.0, 1.0, 1.0, -1.0, 0.5, 3.0]);
    let tall_slope = tape.leaf(slope.clone());
    let wide_slope = tape.leaf(slope.transpose());
    let loss = ((tall * tall_slope).sum() + (wide * wide_slope).sum()).symbol();
    let (tall, wide) = (tall.symbol(), wide.symbol());
    let network = tape.into_network();
    let mut muon = conventional();
    let rate = Tensor::from(1.0_f64);

    let parameters = network.parameters();
    let run = network.forward(&parameters, []);
    let gradients = run.backward(loss).parameters(&parameters);
    let stepped = muon.step(&parameters, &gradients, &rate);

    let tall_update = stepped.of(tall).to_vec();
    let wide_update = stepped.of(wide).transpose().to_vec();
    let rescaled: Vec<f64> = wide_update
        .iter()
        .map(|value| value * 2.0_f64.sqrt())
        .collect();
    assert_close(&tall_update, &rescaled, 1e-12);
}

#[test]
fn higher_rank_parameters_flatten_to_one_row_per_leading_index() {
    // A `[2, 2, 2]` filter bank and the `[2, 4]` matrix holding the
    // same values take bitwise the same update.
    let tape = Tape::new();
    let bank = tape.parameter(Tensor::filled([2, 2, 2], 0.0_f64));
    let matrix = tape.parameter(Tensor::filled([2, 4], 0.0_f64));
    let slope = [2.0_f64, 1.0, 0.0, 1.0, 1.0, -1.0, 0.5, 3.0];
    let bank_slope = tape.leaf(Tensor::new([2, 2, 2], slope));
    let matrix_slope = tape.leaf(Tensor::new([2, 4], slope));
    let loss = ((bank * bank_slope).sum() + (matrix * matrix_slope).sum()).symbol();
    let (bank, matrix) = (bank.symbol(), matrix.symbol());
    let network = tape.into_network();
    let mut muon = conventional();
    let rate = Tensor::from(1.0_f64);

    let parameters = network.parameters();
    let run = network.forward(&parameters, []);
    let gradients = run.backward(loss).parameters(&parameters);
    let stepped = muon.step(&parameters, &gradients, &rate);

    assert_eq!(stepped.of(bank).shape().axes(), [2, 2, 2]);
    assert_bitwise(&stepped.of(bank).to_vec(), &stepped.of(matrix).to_vec());
}

#[test]
fn buffer_covers_parameter_slots_not_graph_nodes() {
    let tape = Tape::new();
    let w = tape.parameter(Tensor::new([2, 2], [1.0_f64, -0.5, 0.25, 2.0]));
    let loss = (w * w).sum().symbol();
    let network = tape.into_network();
    let mut muon = conventional();
    let rate = Tensor::from(0.05_f64);
    let parameters = network.parameters();

    let run = network.forward(&parameters, []);
    let gradients = run.backward(loss).parameters(&parameters);
    muon.step(&parameters, &gradients, &rate);

    let buffer = muon.buffer.as_ref().expect("a step ran");
    assert_eq!(buffer.len(), parameters.len());
    assert!(network.len() > parameters.len());
}

#[test]
fn muon_fits_a_linear_map() {
    // Recover `W` from `X W` under a squared error: the fixed-scale
    // orthogonal steps walk the loss down by orders of magnitude.
    let tape = Tape::new();
    let w = tape.parameter(Tensor::filled([3, 3], 0.0_f64));
    let inputs = Tensor::new(
        [4, 3],
        [
            1.0_f64, -1.0, 0.5, 2.0, 0.5, -1.5, -0.5, 1.5, 1.0, 0.25, -2.0, 0.75,
        ],
    );
    let target = Tensor::new(
        [3, 3],
        [1.0_f64, -0.5, 0.25, 2.0, 0.5, -1.0, 0.75, 1.5, -0.25],
    );
    let x = tape.leaf(inputs.clone());
    let targets = tape.leaf(inputs.matmul(&target));
    let error = x.matmul(w) - targets;
    let loss = (error * error).sum().symbol();
    let network = tape.into_network();
    let mut muon = conventional();

    let mut parameters = network.parameters();
    let initial = network.forward(&parameters, []).of(loss).scalar();
    for step in 0..400 {
        // A geometric decay lets the fixed-size steps settle.
        let rate = Tensor::from(0.1_f64 * 0.99_f64.powi(step));
        let run = network.forward(&parameters, []);
        let gradients = run.backward(loss).parameters(&parameters);
        parameters = muon.step(&parameters, &gradients, &rate);
    }
    let fitted = network.forward(&parameters, []).of(loss).scalar();
    assert!(fitted.is_finite());
    assert!(
        fitted < initial * 1e-3,
        "loss went from {initial} to {fitted}"
    );
}

#[test]
fn pairs_with_adamw_by_selecting_payloads_per_symbol() {
    // The reference recipe: Muon for the hidden matrix, AdamW for the
    // embedding and the bias, each at its own rate, composed as table
    // algebra on the caller's page.
    let tape = Tape::new();
    let embedding = tape.parameter(Tensor::new(
        [4, 2],
        [1.0_f64, -0.5, 0.25, 2.0, 0.5, 1.5, -1.0, 0.75],
    ));
    let hidden = tape.parameter(Tensor::new([2, 2], [0.5_f64, -1.0, 1.5, 0.25]));
    let bias = tape.parameter(Tensor::new([2], [0.5_f64, -1.0]));
    let rows = tape.leaf(Tensor::selection([2, 0, 3], 4, 1.0_f64));
    let features = rows.matmul(embedding).matmul(hidden);
    let shifted = features + bias.broadcast_along_like(0, features);
    let loss = (shifted * shifted).sum();
    let (embedding, hidden, bias, loss) = (
        embedding.symbol(),
        hidden.symbol(),
        bias.symbol(),
        loss.symbol(),
    );
    let network = tape.into_network();

    let mut muon = conventional();
    let mut adamw = AdamW::new(
        Tensor::from(0.9_f64),
        Tensor::from(0.95_f64),
        Tensor::from(1e-8_f64),
        Tensor::from(0.1_f64),
    );
    let muon_rate = Tensor::from(0.02_f64);
    let adamw_rate = Tensor::from(3e-4_f64);

    let parameters = network.parameters();
    let run = network.forward(&parameters, []);
    let gradients = run.backward(loss).parameters(&parameters);
    let by_muon = muon.step_where(&parameters, &gradients, &muon_rate, |symbol, _| {
        symbol == hidden
    });
    let by_adamw = adamw.step(&parameters, &gradients, &adamw_rate);
    let combined: Parameters<f64> = by_muon.step_each(&by_adamw, |symbol, muon, adamw| {
        if symbol == hidden {
            muon.clone()
        } else {
            adamw.clone()
        }
    });

    assert_bitwise(&combined.of(hidden).to_vec(), &by_muon.of(hidden).to_vec());
    assert_bitwise(
        &combined.of(embedding).to_vec(),
        &by_adamw.of(embedding).to_vec(),
    );
    assert_bitwise(&combined.of(bias).to_vec(), &by_adamw.of(bias).to_vec());
}
