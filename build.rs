use cuda_builder::CudaBuilder;

fn main() -> anyhow::Result<()> {
    // Workaround for "crate required to be available in rlib format" bug
    // std::env::set_var("CARGO_BUILD_PIPELINING", "true");

    // Help cargo find libcuda
    // println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/");

    // let builder = Builder::new("core")?;
    // CargoAdapter::with_env_var("KERNEL_PTX_PATH").build(builder);

    CudaBuilder::new("core")
        .copy_to("core.ptx")
        .build()
        .unwrap();

    Ok(())
}