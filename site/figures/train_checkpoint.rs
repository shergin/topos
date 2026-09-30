{
    use topos::{Activation, Mlp, Tape, checkpoint, init};

    let tape: Tape<f32> = Tape::new();
    let mlp = Mlp::new(&tape, &[3, 4, 2], Activation::Relu, init::xavier(11));
    let network = tape.into_network();
    let parameters = network.parameters();

    // A named snapshot is the module's parameters under structured
    // paths; a restore builds a new table from one.
    let snapshot = checkpoint::named_snapshot(&parameters, &mlp);
    let restored = checkpoint::named_restore(&parameters, &mlp, snapshot.clone());
    let mut lines: Vec<String> = snapshot
        .iter()
        .map(|(path, payload)| format!("{path:<12} {}", payload.shape()))
        .collect();
    lines.push(String::new());
    lines.push(format!(
        "restored == original, payload for payload: {}",
        restored.payloads() == parameters.payloads()
    ));
    show::text(lines.join("\n"))
}
