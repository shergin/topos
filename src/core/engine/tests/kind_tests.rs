use crate::{Entry, Numerics, PatternKind, Tape, Tensor, conv2d, max_pool};

/// Records the conv-relu-pool graph the fusion suites use.
fn pooled_network() -> (crate::Network<f64>, crate::Symbol) {
    let tape: Tape<f64> = Tape::new();
    let input = tape.leaf(Tensor::new(
        [2, 1, 4, 4],
        (0..32).map(|v| (v as f64) / 10.0 - 1.5).collect::<Vec<_>>(),
    ));
    let weights = tape.leaf(Tensor::new(
        [2, 1, 3, 3],
        (0..18)
            .map(|v| (v as f64) / 24.0 - 0.375)
            .collect::<Vec<_>>(),
    ));
    let bias = tape.leaf(Tensor::new([2], [0.05, -0.05]));
    let output = max_pool(conv2d(input, weights, bias, 1, 1).relu(), 2, 2).symbol();
    (tape.into_network(), output)
}

#[test]
fn elected_patterns_are_data() {
    let (network, output) = pooled_network();
    let plan = network.entry([output]).lower();

    let patterns = plan.patterns();
    let kinds: Vec<PatternKind> = patterns.iter().map(|group| group.kind()).collect();
    assert!(kinds.contains(&PatternKind::WindowProduct));
    assert!(kinds.contains(&PatternKind::ReduceWindow));

    // Elected groups come from the candidate pool: every elected root
    // appears among the candidates, and every claimed node includes
    // its own root.
    let candidate_roots: Vec<_> = plan.candidates().iter().map(|group| group.root()).collect();
    for group in &patterns {
        assert!(candidate_roots.contains(&group.root()));
        assert!(group.nodes().contains(&group.root()));
        assert!(group.nodes().len() > 1);
    }
}

#[test]
fn engine_backward_plans_elect_nothing_but_still_see_the_pool() {
    let (network, output) = pooled_network();
    let plan = network.compile(Entry::roots([output]).backward());
    assert!(plan.patterns().is_empty());
    assert!(!plan.candidates().is_empty());
}

#[test]
fn describe_agrees_with_the_data() {
    let (network, output) = pooled_network();
    let plan = network.entry([output]).lower();
    let described = plan.describe();
    for group in plan.patterns() {
        // The root's spec line is printed, and the root's own line is
        // where the fused group's result lives.
        let line = plan.node(group.root()).to_string();
        assert!(
            described.contains(line.trim_end()),
            "describe is missing the elected root line {line:?}"
        );
    }
}

#[test]
fn exact_numerics_still_elects_the_bit_certified_kernels() {
    // The window kernels are bit-certified, so the Exact posture keeps
    // them; the formula-to-fidelity admission is per pattern, not a
    // blanket refusal.
    let (network, output) = pooled_network();
    let plan = network.entry([output]).numerics(Numerics::Exact).lower();
    let interpreted = network.forward(&network.parameters(), []);
    let planned = plan.forward(&network.parameters(), []);
    assert_eq!(planned.of(output).to_vec(), interpreted.of(output).to_vec());
}
