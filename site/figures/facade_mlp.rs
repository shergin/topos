{
    use topos::{Activation, Mlp, Module, Tape, Tensor, init};

    let tape: Tape<f32> = Tape::new();
    let mlp = Mlp::new(&tape, &[2, 4, 1], Activation::Tanh, init::xavier(7));
    let _output = mlp.express(tape.input(Tensor::filled([8, 2], 0.0_f32)));
    show::graph(Graph::from_nodes(tape.nodes()))
}
