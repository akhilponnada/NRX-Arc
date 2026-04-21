# Project: NRX-Arc

## What this project is

A from-scratch ML training framework in Rust + CUDA with a Python frontend. 
Built to train a ~500M parameter modern decoder-only transformer on a single 
NVIDIA GH200 (96GB HBM3, CUDA 12.8, Hopper sm_90). Inspired by Aadi 
Kulshrestha's mni-ml/framework (TypeScript + 12M vanilla transformer), but 
deliberately different and better on specific axes.

## Why this project exists (keep this front-of-mind)

I am a solo builder with ~10 weeks to ship this. The goal is a high-quality 
educational artifact with enough depth and polish to credibly demonstrate 
end-to-end systems skills. Not a production framework. Not novel research.

## Differentiators vs. Aadi's framework (locked, do not second-guess)

These are the axes on which this project will be better than mni-ml/framework. 
If a design decision trades one of these away, flag it before proceeding.

1. **Python API**, not TypeScript. Python is where ML actually lives. PyO3 
   bindings, `pip install` path.
2. **Modern architecture**: RoPE, RMSNorm, SwiGLU, grouped-query attention, 
   tied embeddings. No absolute positional embeddings. No vanilla LayerNorm 
   in the transformer blocks. Aadi's was 2019-era; ours is 2024-era.
3. **500M parameters**, not 12M. ~40× bigger. A real small model, not a toy.
4. **BF16 mixed precision training** with FP32 master weights. FP8 (E4M3) 
   training as a Phase 7 stretch goal, gated on BF16 working cleanly. Aadi 
   is fp32-only.
5. **Honest published benchmarks** against a PyTorch + FlashAttention-2 
   baseline, op by op. We will be slower. We publish the ratio anyway. 
   Aadi publishes no benchmarks.
6. **Real evaluation** against Pythia-410M and OLMo-1B early checkpoints 
   on HellaSwag, PIQA, LAMBADA. Aadi has zero eval numbers.
7. **Gradient-checked test suite** — every op is numerically validated 
   against PyTorch reference to 1e-3 relative error. Aadi has tests but 
   not systematic numerical validation.
8. **Live attention visualization demo** — web demo where user types a 
   prompt, sees generation token-by-token with attention heatmaps over 
   selected layers and heads, next-token probability distribution. Core 
   deliverable, not optional. Aadi's demo shows probabilities only.
9. **Blog quality**. 4-5 posts. This is the single highest-leverage 
   differentiator. Invest real time in weeks 9-10.

## What this project explicitly IS NOT

- Not faster than PyTorch. Target is 40-60% of a good PyTorch + FA2 
  baseline. Don't claim otherwise anywhere in code comments, docs, 
  or the eventual blog.
- Not a production framework. No distributed training, no FSDP, no ZeRO, 
  no tensor parallelism.
- Not novel research. No new architectures, optimizers, or training 
  techniques. We're executing known-good techniques cleanly.
- Not a general-purpose ML framework. Transformer-focused. We do not 
  need Conv1d/Conv2d/pooling/etc. that Aadi's framework has.
- Not a browser-native inference runtime. Aadi's WebGPU backend is his 
  win; we skip it. Browser demo uses a hosted inference endpoint.

## Anti-patterns to avoid (do not propose these)

- Do NOT suggest adding a WebGPU backend. We explicitly skip this.
- Do NOT suggest adding conv1d/conv2d/pooling. Transformer-only scope.
- Do NOT suggest a TypeScript API. Python-only.
- Do NOT try to match cuBLAS/cuDNN performance. We benchmark against 
  them honestly, we don't try to beat them.
- Do NOT propose novel architectures, optimizers, or training techniques.
- Do NOT suggest distributed training. Single-GPU only.
- Do NOT suggest TMA/WGMMA/FlashAttention-3 kernels. Too hard in 10 weeks. 
  FA2 is the ceiling.
- Do NOT suggest ONNX export or browser-native inference. Hosted endpoint 
  is how the demo works.
- Do NOT add marketing language to code or docstrings. No "blazing fast", 
  "optimized", "state-of-the-art". Say what the code does.

## Non-negotiable correctness rules

This project has one specific failure mode: CUDA kernels and backward 
passes that compile, run, produce numbers, and are subtly wrong. We do 
not let this happen.

1. **Every custom op gets a gradient check before we move on.** For any op 
   with a backward pass, write a test in `tests/grad_check.py` that 
   compares analytic gradients to finite-difference numerical gradients 
   (relative tolerance 1e-3 for fp32, 1e-2 for bf16). If the op exists 
   in PyTorch, also compare forward output directly.
2. **No "it works" without a passing test.** Feature is not done until 
   the test is written and passing.
3. **Every CUDA kernel gets benchmarked against a reference** — cuBLAS, 
   cuDNN, or PyTorch. We need to know the ratio. If ours is >3× slower, 
   flag it.
4. **Numerical stability is not optional**: softmax with max-subtraction, 
   LayerNorm/RMSNorm stats in FP32 even when inputs are BF16, loss 
   scaling for FP8. Document these choices in comments.
5. **Never silently change a sign, axis, or dtype.** If you need to, say 
   so in a comment explaining why.

## Tech stack (decided, do not re-debate)

- Rust 2021, stable toolchain
- CUDA 12.8 (installed on target GH200)
- Python 3.12 bindings via PyO3 0.23 + maturin
- cudarc 0.19.4 for CUDA device management (NOT 0.16; the API changed —
  CudaContext singleton, Stream-based alloc/copy, launch_builder for kernels).
  Feature flags: `cuda-12080` (matches our CUDA 12.8 toolkit) +
  `dynamic-loading` (lets the .so resolve libcuda at load time, not link time).
- Custom .cu kernels compiled via NVCC, loaded as PTX
- BF16 first (Phase 5), FP8 E4M3 stretch (Phase 7)
- We write our own matmul, softmax, layernorm/rmsnorm, attention as 
  learning exercises. We use cuBLAS as a benchmark reference only.

## Target hardware

- NVIDIA GH200 480GB (Grace Hopper)
- 96GB HBM3, 525GB unified LPDDR5X (ignore the unified memory — we 
  decided not to frame the project around it)
- sm_90 (Hopper), not sm_90a. No TMA/WGMMA.
- CUDA driver 580.105.08

## Phase plan (10 weeks, adjusted for differentiators)

Currently in: **Phase 2 — CUDA port. Phase 1 complete (tag `phase-1-complete`).**

- **Phase 1 (week 1)**: Rust tensor engine, CPU only, autograd, PyO3 
  bindings, MLP on synthetic data
- **Phase 2 (week 2)**: Port to CUDA. Elementwise ops, matmul, gradient- 
  check everything.
- **Phase 3 (week 3)**: Transformer ops — RMSNorm, softmax, naive 
  attention, embedding, cross-entropy, AdamW. Modern-arch from the 
  start (RMSNorm, not LayerNorm).
- **Phase 4 (weeks 4-5)**: Modern transformer — RoPE, SwiGLU MLP, GQA, 
  Flash Attention v2 forward + backward.
- **Phase 5 (week 6)**: BF16 mixed precision. Master weights FP32, 
  compute BF16, loss scaling, careful upcasts.
- **Phase 6 (week 7)**: Data pipeline, BPE tokenizer, training config, 
  launch the 500M training run on FineWeb-Edu subset (~20-40B tokens).
- **Phase 7 (week 8)**: FP8 stretch goal IF BF16 run succeeded. If not, 
  skip FP8 entirely, spend the week on evaluation + benchmarks.
- **Phase 8 (week 9)**: Attention-viz demo (hosted inference endpoint, 
  browser frontend), evaluation vs Pythia-410M / OLMo-1B.
- **Phase 9 (week 10)**: Blog posts (4-5 posts), launch materials, 
  repo cleanup, launch thread.

If we fall behind, cut in this order: (a) FP8, (b) one blog post, 
(c) drop from 500M to 350M. Do not cut the gradient checks, the 
benchmarks, the evals, or the demo.

## Current state

Phase 1 shipped: CPU autograd, 8 ops (add, mul, matmul, relu, sum, neg,
sub, scalar_mul), nn.Module + Parameter + Linear, optim.SGD, MLP at 95%
test acc on 3-blob synthetic data. All 24 tests passing. Tagged
`phase-1-complete`. See DEVLOG.md for the full writeup.

Phase 2 in progress: porting the 8 CPU ops to CUDA on the GH200. Do not
assume any CUDA code exists unless you've `view`'d it.

## Phase 2 design (locked decisions, do not re-debate)

These were settled before Phase 2 step 1 began. If a future change wants
to revisit any of these, flag it explicitly rather than silently drift.

- **cudarc 0.19.4** with features `cuda-12080` + `dynamic-loading`. Pinned
  in Cargo.toml. Earlier plan said 0.16 — that was stale.
- **Storage enum on Tensor**, not generic-over-device. One `Tensor` type,
  internally `Storage::Cpu(Vec<f32>)` or `Storage::Cuda(CudaSlice<f32>)`.
  Same for `grad`. Generics-over-device would force a parallel type for
  every op; the enum keeps the public API a single shape.
- **`.to(device)` returns a new tensor**, never mutates in place. In-place
  device transfers create tape-vs-storage hazards we don't need yet.
- **`CudaContext` is a process-wide singleton via `OnceLock<Arc<CudaContext>>`**.
  `Storage::Cuda` holds a `CudaSlice<f32>` only — NO per-tensor
  `Arc<CudaContext>`. The singleton is `'static`, so the slice is always
  valid. A warning comment on the `Cuda` variant explains this so a future
  collaborator doesn't "fix" it by adding the Arc back.
- **Hard-fail if `nvcc` is missing.** No optional CUDA build. `build.rs`
  errors out with a clear "install CUDA 12.x toolkit, ensure nvcc on PATH"
  message. The whole point of this project is the GPU path.
- **Mixed-device ops raise `PyRuntimeError`** with both device names in the
  message (e.g. `"cannot add tensors on different devices: cpu and cuda:0"`).
  No silent host↔device copies.
- **Device string format is `"cpu"` and `"cuda:0"`**, matching PyTorch.
- **Tape's `accumulate_grad` does a `debug_assert!` for device match** between
  the tensor's storage and the incoming grad. Silent wrong-device grad
  accumulation would be brutal to debug; the assert catches it in dev
  without paying for it in release.
- **Phase 2 commit-by-commit plan**:
  1. Storage enum, `.to()` round-trip, mixed-device errors, device tests.
     No `.cu` files yet. All 8 ops still CPU-only; CUDA paths are
     `unimplemented!()`. 24 existing tests + 4 device tests pass.
  2. First real kernel: elementwise `add` on CUDA, with CPU-vs-CUDA
     equivalence test + FD grad check on the GPU.
  3. Remaining elementwise ops, then matmul (naive first, benchmarked
     against cuBLAS), then sum/relu, then MLP integration test on GPU.

## When you're unsure

- Ask. Do not make up APIs. If you don't know a cudarc 0.19 signature, 
  `view` the docs or the crate source before writing code against it.
- If a test is failing weirdly, don't guess-and-check. Read the numbers, 
  form a hypothesis, verify. CUDA bugs punish guess-and-check.
- Never claim a kernel is "numerically identical to PyTorch" without 
  running a comparison and showing me the max absolute diff.

## Honesty rules for code comments and docstrings

- No "blazing fast", "optimized", "state-of-the-art", "novel", "paradigm"
- Say what the code does and what its limitations are
- If our matmul is 2.5× slower than cuBLAS, the comment says exactly that
- Backward passes that involve tricky transposes get a comment explaining 
  the derivation

  # Git workflow for this project

## Two repos, not one

1. **Code repo** (private until launch): `nrx-arc`.
   All the Rust/CUDA/Python code.
2. **Blog repo** (private until launch): `nrx-arc-blog`. 
   Separate repo, deployed via GitHub Pages. Markdown posts + a minimal 
   static site generator (we'll use Zola or 11ty — decided in Phase 9).

Keeping them separate means:
- You can flip the blog public before the code repo, or vice versa, for 
  launch choreography
- Blog iteration doesn't pollute code commit history
- Aadi does it this way (mni-ml.github.io is a separate repo from 
  mni-ml/framework)

## Initial setup on the GH200

Run this once, on the GH200, in your home directory:

```bash