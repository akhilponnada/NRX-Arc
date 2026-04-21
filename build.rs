// Build script: hard-fail if nvcc isn't on PATH. nrx-arc is a Rust + CUDA
// framework — there's no point compiling without CUDA tooling, and silent
// fallback to CPU-only would just defer the eventual link error to runtime.
//
// Future commits will scan `cuda/*.cu` here, compile each to PTX with
// `nvcc -arch=sm_90 -O2`, and place the .ptx files where `cudarc::nvrtc`
// or `load_module` can find them. There are no .cu files yet.
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-changed=build.rs");

    match Command::new("nvcc").arg("--version").output() {
        Ok(out) if out.status.success() => {
            // nvcc is on PATH — good.
        }
        _ => {
            panic!(
                "nrx-arc requires CUDA 12.x. Install the CUDA toolkit and ensure nvcc is on \
                 PATH (CUDA_HOME=/usr/local/cuda is typical). `nvcc --version` failed."
            );
        }
    }
}
