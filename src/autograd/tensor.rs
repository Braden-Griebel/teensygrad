// Standard Library Use statements
use std::cell::RefCell;
use std::sync::Arc;

// External Crate Uses
use nalgebra::DMatrix;

/// A wrapped tensor, which will track computations
/// enabling automatic differentiation
pub struct Tensor {
    inner: Arc<RefCell<InnerTensor>>,
}

/// Internal Tensor
struct InnerTensor {
    /// The operation which created the Tensor
    op: Operation,
    /// The value contained in the Tensor
    value: DMatrix<f64>,
    /// The gradient of the Tensor
    grad: DMatrix<f64>,
    /// The predecessors of the Tensor (Tensors involved in the calculation creating this tensor)
    predecessors: TensorPredecessor,
}

impl InnerTensor{
    /// Calculate the gradient of the Tensor during backpropagation
    fn backwards(){
        todo!();
    }
}

enum TensorPredecessor {
    None,
    One(Arc<RefCell<InnerTensor>>),
    Two{
        left: Arc<RefCell<InnerTensor>>,
        right: Arc<RefCell<InnerTensor>>,
    }
}

/// Enum describing the operation that created a tensor
enum Operation {
    Add,
    Sub,
    Sum(Axis),
    Pow(f64),
    MatMul,
}

/// Axis along which an operation was performed
enum Axis {
    Row,
    Col,
}