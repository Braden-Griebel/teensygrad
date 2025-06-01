// Standard Library Use statements
use std::cell::RefCell;
use std::rc::Rc;

// External Crate Uses
use ndarray::{ArrayD, Axis};

/// A wrapped tensor, which will track computations
/// enabling automatic differentiation
pub struct Tensor {
    inner: Rc<RefCell<InnerTensor>>,
}

/// Internal Tensor
struct InnerTensor {
    /// The operation which created the Tensor
    op: Operation,
    /// The value contained in the Tensor
    value: ArrayD<f64>,
    /// The gradient of the Tensor
    grad: ArrayD<f64>,
    /// The predecessors of the Tensor (Tensors involved in the calculation creating this tensor)
    predecessors: TensorPredecessor,
}

impl InnerTensor{
    /// Calculate the gradient of the Tensor during backpropagation
    fn backwards(){
        todo!();
    }
}

/// An enum to describe the possible predecessors of a Tensor
enum TensorPredecessor {
    /// If the Tensor has no predecessors (i.e. was directly created rather than from a calculation)
    None,
    /// If the Tensor was created by a unary operation
    One(Rc<RefCell<InnerTensor>>),
    /// If the Tensor was created by a Binary Operation
    Two{
        left: Rc<RefCell<InnerTensor>>,
        right: Rc<RefCell<InnerTensor>>,
    }
}

/// Enum describing the operation that created a tensor
enum Operation {
    Add,
    Concat {axis: Axis, size: isize},
    Conv2D {pad: Option<f64>, stride: isize}, 
    Cosh, 
    Mult,
    MatMult,
    Pow(f64),
    Split {axis: Axis, size: isize},
    Sub,
    Sum(Axis),
}