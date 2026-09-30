{
    use std::panic::{self, AssertUnwindSafe};

    use topos::{Tape, Tensor};

    let tape: Tape<f32> = Tape::new();
    let left = tape.input(Tensor::filled([3, 2], 0.0_f32));
    let right = tape.input(Tensor::filled([4, 5], 0.0_f32));

    // A `[3, 2]` by `[4, 5]` product has no shape; the recording line
    // itself panics, and this page catches it to print the message.
    let quiet = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let outcome = panic::catch_unwind(AssertUnwindSafe(|| left.matmul(right)));
    panic::set_hook(quiet);
    let message = match outcome {
        Ok(_) => "no panic".to_string(),
        Err(payload) => payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|text| text.to_string()))
            .unwrap_or_default(),
    };
    show::lines([
        "left.matmul(right) panicked at the recording expression:".to_string(),
        format!("  {message}"),
    ])
}
