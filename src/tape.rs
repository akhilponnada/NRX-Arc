use pyo3::prelude::*;
use std::cell::RefCell;
use crate::tensor::Tensor;
use crate::device::{Device, Storage};

#[derive(Clone, Copy, Debug)]
pub enum BroadcastKind {
    SameShape,
    LeftScalar,
    RightScalar,
    /// a is 2D [rows, cols], b is 1D [cols]. Used for `out + bias`.
    RowVecRight { rows: usize, cols: usize },
    /// a is 1D [cols], b is 2D [rows, cols]. Symmetric of the above.
    RowVecLeft { rows: usize, cols: usize },
}

pub enum OpKind {
    Add { broadcast: BroadcastKind },
    Mul { broadcast: BroadcastKind },
    Matmul { m: usize, k: usize, n: usize },
    Relu,
    Sum { input_numel: usize },
    Neg,
    ScalarMul { s: f32 },
}

pub struct OpRecord {
    pub kind: OpKind,
    pub inputs: Vec<Py<Tensor>>,
    pub output: Py<Tensor>,
}

pub struct Tape { pub records: Vec<OpRecord> }
impl Tape { pub fn new() -> Self { Tape { records: Vec::new() } } }

// PHASE 1: single-threaded CPU is fine; revisit for multi-device later.
thread_local! {
    pub static TAPE: RefCell<Tape> = RefCell::new(Tape::new());
}

pub fn record(rec: OpRecord) {
    TAPE.with(|t| t.borrow_mut().records.push(rec));
}

pub fn clear() {
    TAPE.with(|t| t.borrow_mut().records.clear());
}

/// Accumulate a host-side gradient contribution into a tensor's `.grad`.
///
/// In Phase 2 step 1 every backward pass still runs on CPU, so `contrib` is a
/// host slice and the receiving tensor must be on CPU too. The debug_assert
/// catches the case where someone wires a CUDA tensor into a CPU backward
/// without realizing it — silent wrong-device grad accumulation would be
/// brutal to debug.
pub fn accumulate_grad(py: Python, t: &Py<Tensor>, contrib: &[f32]) {
    let mut t_ref = t.borrow_mut(py);
    debug_assert_eq!(
        t_ref.storage.device(),
        Device::Cpu,
        "accumulate_grad: backward currently only supports CPU tensors"
    );
    match &mut t_ref.grad {
        None => t_ref.grad = Some(Storage::Cpu(contrib.to_vec())),
        Some(Storage::Cpu(g)) => {
            assert_eq!(g.len(), contrib.len(), "grad shape mismatch in accumulate");
            for (gi, ci) in g.iter_mut().zip(contrib.iter()) { *gi += *ci; }
        }
        Some(Storage::Cuda(_)) => {
            unimplemented!("cuda grad accumulation not yet")
        }
    }
}

pub fn backward(py: Python, loss: Py<Tensor>) -> PyResult<()> {
    // Guard against double-backward and backward-on-leaf. Tape is the
    // authoritative signal: an empty tape means there's nothing to walk.
    let tape_empty = TAPE.with(|t| t.borrow().records.is_empty());
    if tape_empty {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "backward() called but tape is empty - did you already call backward on this graph? \
             Each graph can only be backpropagated once; forward again to build a new graph."
        ));
    }
    {
        let mut loss_ref = loss.borrow_mut(py);
        if loss_ref.grad.is_none() {
            let n = loss_ref.shape.iter().product::<usize>().max(1);
            loss_ref.grad = Some(Storage::Cpu(vec![1.0; n]));
        }
    }
    let records: Vec<OpRecord> = TAPE.with(|t| std::mem::take(&mut t.borrow_mut().records));
    for rec in records.into_iter().rev() {
        match rec.kind {
            OpKind::Add { broadcast } => crate::ops::add::backward(py, &rec, broadcast)?,
            OpKind::Mul { broadcast } => crate::ops::mul::backward(py, &rec, broadcast)?,
            OpKind::Matmul { m, k, n } => crate::ops::matmul::backward(py, &rec, m, k, n)?,
            OpKind::Relu => crate::ops::relu::backward(py, &rec)?,
            OpKind::Sum { input_numel } => crate::ops::sum::backward(py, &rec, input_numel)?,
            OpKind::Neg => crate::ops::neg::backward(py, &rec)?,
            OpKind::ScalarMul { s } => crate::ops::scalar_mul::backward(py, &rec, s)?,
        }
    }
    Ok(())
}

/// Helper used by op backward passes to read an output tensor's grad as a
/// host vec. Asserts the grad lives on CPU (Phase 2 step 1 invariant).
pub fn output_grad_cpu(py: Python, t: &Py<Tensor>, op_name: &str) -> Vec<f32> {
    let t_ref = t.borrow(py);
    match &t_ref.grad {
        Some(Storage::Cpu(v)) => v.clone(),
        Some(Storage::Cuda(_)) => unimplemented!("cuda backward not yet ({})", op_name),
        None => panic!("{} backward: missing output grad", op_name),
    }
}

/// Helper used by op backward passes to read an input tensor's data as a
/// host vec. Asserts the data lives on CPU.
pub fn input_data_cpu(py: Python, t: &Py<Tensor>, op_name: &str) -> Vec<f32> {
    let t_ref = t.borrow(py);
    match &t_ref.storage {
        Storage::Cpu(v) => v.clone(),
        Storage::Cuda(_) => unimplemented!("cuda backward not yet ({})", op_name),
    }
}
