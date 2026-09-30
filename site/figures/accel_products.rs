{
    use malevich::{Bars, Plot};

    // The measured table on this page: dense f32 products, GFLOP/s,
    // one Apple M1 Pro.
    let builds = ["default", "simd", "accelerate", "metal"];
    let gflops = [26.0, 96.0, 1600.0, 1400.0];
    show::plot(
        Plot::new()
            .layer(Bars::new(builds, gflops).horizontal())
            .title("dense f32 products by build")
            .x_label("GFLOP/s")
            .log_x(),
    )
}
