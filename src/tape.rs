use pyo3::prelude::*;
use std::cell::RefCell;
use crate::tensor::Tensor;

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

pub fn accumulate_grad(py: Python, t: &Py<Tensor>, contrib: &[f32]) {
    let mut t_ref = t.borrow_mut(py);
    match &mut t_ref.grad {
        None => t_ref.grad = Some(contrib.to_vec()),
        Some(g) => {
            assert_eq!(g.len(), contrib.len(), "grad shape mismatch in accumulate");
            for (gi, ci) in g.iter_mut().zip(contrib.iter()) { *gi += *ci; }
        }
    }
}

pub fn backward(py: Python, loss: Py<Tensor>) -> PyResult<()> {
    // Guard against double-backward and backward-on-leaf. Tape is the
    // authoritative signal: an empty tape means there's nothing to walk.
    let tape_empty = TAPE.with(|t| t.borrow().records.is_empty());
    if tape_empty {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "backward() called but tape is empty â did you already call backward on this graph?              Each graph can only be backpropagated once; forward again to build a new graph."
        ));
    }
    {
        let mut loss_ref = loss.borrow_mut(py);
        if loss_ref.grad.is_none() {
            let n = loss_ref.shape.iter().product::<usize>().max(1);
            loss_ref.grad = Some(vec![1.0; n]);
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
