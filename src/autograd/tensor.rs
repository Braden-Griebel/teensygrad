// Standard Library Use statements
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

// External Crate Uses
use anyhow::{Context, Result};
use ndarray::{ArrayD, ArrayViewD, Axis, Zip};
use ndarray_rand::rand_distr::{Distribution, Normal, Uniform};
use ndarray_rand::RandomExt;
use thiserror;

/// A wrapped tensor, which will track computations
/// enabling automatic differentiation
pub struct Tensor {
    inner: Rc<RefCell<InnerTensor>>,
}

// Implements the representation for the Tensor
// (Just uses the display method of the underlying ndarray)
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.inner.borrow().value)
    }
}

// Functions for creating Tensors
impl Tensor {
    /// Create a new Tensor with shape `shape`, filled with zeros
    fn new_zeros(shape: &[usize]) -> Result<Self> {
        Ok(Self {
            inner: Rc::new(RefCell::new(
                InnerTensor::new_zeros(shape).context("Failed to create inner tensor")?,
            )),
        })
    }

    /// Create a new Tensor with shape `shape`, filled with ones
    fn new_ones(shape: &[usize]) -> Result<Self> {
        Ok(Self {
            inner: Rc::new(RefCell::new(
                InnerTensor::new_ones(shape).context("Failed to create inner tensor")?,
            )),
        })
    }
    /// Create a new Tensor with shape `shape`, filled with `elem`
    fn new_from_elem(shape: &[usize], elem: f64) -> Result<Self> {
        Ok(Self {
            inner: Rc::new(RefCell::new(
                InnerTensor::new_from_elem(shape, elem)
                    .context("Failed to initialize an inner tensor with elem")?,
            )),
        })
    }

    /// Create a new Tensor with shape `shape`, filled with
    /// values sampled from a uniform distribution between 0 and 1
    fn new_rand(shape: &[usize]) -> Result<Self> {
        Ok(Self {
            inner: Rc::new(RefCell::new(
                InnerTensor::new_rand(shape)
                    .context("Failed to create inner tensor with uniform random values")?,
            )),
        })
    }

    /// Create a new Tensor with shape `shape`, filled with
    /// values sampled from a normal distribution with mean 0 and
    /// standard deviation 1
    fn new_rand_norm(shape: &[usize]) -> Result<Self> {
        Ok(Self {
            inner: Rc::new(RefCell::new(InnerTensor::new_rand_norm(shape).context(
                "Failed to create inner tensor with values from normal distribution",
            )?)),
        })
    }
}

// Functions to get information about the Tensor
impl Tensor {
    // Passthrough functions for the Value
    /// Determine the number of dimensions the Tensor has
    fn ndim(&self) -> usize {
        self.inner.borrow().ndim()
    }

    /// Determine the number of elements the Tensor has
    fn len(&self) -> usize {
        self.inner.borrow().len()
    }

    /// Determine the shape of the Tensor
    fn shape(&self) -> Vec<usize> {
        self.inner.borrow().shape().to_owned()
    }

    /// Determine whether the Tensor has any elements
    fn is_empty(&self) -> bool {
        self.inner.borrow().is_empty()
    }

    // Passthrough functions for the grad

    /// Determine the number of dimensions the gradient array has
    fn grad_ndim(&self) -> usize {
        self.inner.borrow().grad_ndim()
    }

    /// Determine the number of elements the gradient array has
    fn grad_len(&self) -> usize {
        self.inner.borrow().grad_len()
    }

    /// Determine the shape of the gradient array
    fn grad_shape(&self) -> Vec<usize> {
        self.inner.borrow().grad_shape().to_owned()
    }

    /// Determine whether the grad has any elements
    fn grad_is_empty(&self) -> bool {
        self.inner.borrow().grad_is_empty()
    }
}

// Functions to view and modify the values associated with the Tensor
impl Tensor {
    /// Set the value of the gradient for the Tensor to 0
    fn zero_grad(&mut self) {
        self.inner.borrow_mut().zero_grad();
    }

    /// Get a copy of the value represented by the Tensor
    fn get_value_copy(&self) -> ArrayD<f64> {
        let inner = self.inner.borrow();
        inner.value.clone()
    }

    /// Get a copy of the gradient represented by the Tensor
    fn get_grad_copy(&self) -> ArrayD<f64> {
        let inner = self.inner.borrow();
        inner.grad.clone()
    }
}

// Operations on Tensors
impl Tensor {
    // SECTION: Unary Operations
    /// Element-wise hyperbolic cosine
    fn cos(input: Tensor) -> Result<Self> {
        let result = input.inner.borrow().value.map(|x| x.cos());
        let result_shape: Vec<usize> = result.shape().to_vec();
        Ok(Self {
            inner: Rc::new(RefCell::new(InnerTensor {
                op: Operation::Cos,
                value: result,
                grad: ArrayD::zeros(result_shape),
                predecessors: TensorPredecessor::One(input.inner.clone()),
            })),
        })
    }

    /// Element-wise hyperbolic cosine
    fn cosh(input: Tensor) -> Result<Self> {
        let result = input.inner.borrow().value.map(|x| x.cosh());
        let result_shape: Vec<usize> = result.shape().to_vec();
        Ok(Self {
            inner: Rc::new(RefCell::new(InnerTensor {
                op: Operation::Cosh,
                value: result,
                grad: ArrayD::zeros(result_shape),
                predecessors: TensorPredecessor::One(input.inner.clone()),
            })),
        })
    }

    /// Element-wise exponential base e
    fn exp(input: Tensor) -> Result<Self> {
        let result = input.inner.borrow().value.map(|x| x.exp());
        let result_shape: Vec<usize> = result.shape().to_vec();
        Ok(Self {
            inner: Rc::new(RefCell::new(InnerTensor {
                op: Operation::Exp,
                value: result,
                grad: ArrayD::zeros(result_shape),
                predecessors: TensorPredecessor::One(input.inner.clone()),
            })),
        })
    }

    /// Element-wise exponential with base `base`
    fn exp_base(input: Tensor, base: f64) -> Result<Self> {
        let result = input.inner.borrow().value.map(|x| base.powf(*x));
        let result_shape: Vec<usize> = result.shape().to_vec();
        Ok(Self {
            inner: Rc::new(RefCell::new(InnerTensor {
                op: Operation::ExpBase(base),
                value: result,
                grad: ArrayD::zeros(result_shape),
                predecessors: TensorPredecessor::One(input.inner.clone()),
            })),
        })
    }

    /// Element-wise Heavide function (converts elements to 0 below
    ///  `location`, and 1 above `location`)
    fn heaviside(input: Tensor, location: f64) -> Result<Self> {
        let result = input
            .inner
            .borrow()
            .value
            .map(|x| if *x < location { 0f64 } else { 1f64 });
        let result_shape: Vec<usize> = result.shape().to_vec();
        Ok(Self {
            inner: Rc::new(RefCell::new(InnerTensor {
                op: Operation::Heaviside(location),
                value: result,
                grad: ArrayD::zeros(result_shape),
                predecessors: TensorPredecessor::One(input.inner.clone()),
            })),
        })
    }

    /// Element-wise power of Tensor
    fn pow(input: Tensor, exponent: f64) -> Result<Self> {
        let result = input.inner.borrow().value.map(|x| x.powf(exponent));
        let result_shape: Vec<usize> = result.shape().to_vec();
        Ok(Self {
            inner: Rc::new(RefCell::new(InnerTensor {
                op: Operation::Pow(exponent),
                value: result,
                grad: ArrayD::zeros(result_shape),
                predecessors: TensorPredecessor::One(input.inner.clone()),
            })),
        })
    }

    /// Element-wise Rectified Linear Unit function
    fn relu(input: Tensor) -> Result<Self> {
        let result = input
            .inner
            .borrow()
            .value
            .map(|x| if *x < 0f64 { 0f64 } else { *x });
        let result_shape: Vec<usize> = result.shape().to_vec();
        Ok(Self {
            inner: Rc::new(RefCell::new(InnerTensor {
                op: Operation::Relu,
                value: result,
                grad: ArrayD::zeros(result_shape),
                predecessors: TensorPredecessor::One(input.inner.clone()),
            })),
        })
    }

    /// Element-wise hyperbolic cosine
    fn sin(input: Tensor) -> Result<Self> {
        let result = input.inner.borrow().value.map(|x| x.sin());
        let result_shape: Vec<usize> = result.shape().to_vec();
        Ok(Self {
            inner: Rc::new(RefCell::new(InnerTensor {
                op: Operation::Sin,
                value: result,
                grad: ArrayD::zeros(result_shape),
                predecessors: TensorPredecessor::One(input.inner.clone()),
            })),
        })
    }
    /// Element-wise hyperbolic cosine
    fn sinh(input: Tensor) -> Result<Self> {
        let result = input.inner.borrow().value.map(|x| x.sinh());
        let result_shape: Vec<usize> = result.shape().to_vec();
        Ok(Self {
            inner: Rc::new(RefCell::new(InnerTensor {
                op: Operation::Sinh,
                value: result,
                grad: ArrayD::zeros(result_shape),
                predecessors: TensorPredecessor::One(input.inner.clone()),
            })),
        })
    }

    /// Split a Tensor along `axis`, at position `index`
    fn split(input: Self, axis: Axis, index: usize) -> Result<(Self, Self)> {
        let inner_tensor = input.inner.borrow();
        let (left, right) = inner_tensor.value.view().split_at(axis, index);
        let left_shape = left.shape().to_vec();
        let right_shape = right.shape().to_vec();
        return Ok((
            Self {
                inner: Rc::new(RefCell::new(InnerTensor {
                    value: left.clone().to_owned(),
                    grad: ArrayD::zeros(left_shape),
                    op: Operation::Split {
                        axis,
                        index,
                        first: true,
                    },
                    predecessors: TensorPredecessor::One(input.inner.clone()),
                })),
            },
            Self {
                inner: Rc::new(RefCell::new(InnerTensor {
                    value: right.clone().to_owned(),
                    grad: ArrayD::zeros(right_shape),
                    op: Operation::Split {
                        axis,
                        index,
                        first: false,
                    },
                    predecessors: TensorPredecessor::One(input.inner.clone()),
                })),
            },
        ));
    }

    /// Element-wise Tangent function
    fn tan(input: Tensor) -> Result<Self> {
        let result = input.inner.borrow().value.map(|x| x.tan());
        let result_shape: Vec<usize> = result.shape().to_vec();
        Ok(Self {
            inner: Rc::new(RefCell::new(InnerTensor {
                op: Operation::Tan,
                value: result,
                grad: ArrayD::zeros(result_shape),
                predecessors: TensorPredecessor::One(input.inner.clone()),
            })),
        })
    }

    /// Element-wise Hyperbolic Tangent function
    fn tanh(input: Tensor) -> Result<Self> {
        let result = input.inner.borrow().value.map(|x| x.tanh());
        let result_shape: Vec<usize> = result.shape().to_vec();
        Ok(Self {
            inner: Rc::new(RefCell::new(InnerTensor {
                op: Operation::Tanh,
                value: result,
                grad: ArrayD::zeros(result_shape),
                predecessors: TensorPredecessor::One(input.inner.clone()),
            })),
        })
    }

    // SECTION: Binary Operations
    /// Element-wise addition of two Tensors
    fn add(left: Self, right: Self) -> Result<Self> {
        let result = &left.inner.borrow().value + &right.inner.borrow().value;
        let result_shape: Vec<usize> = result.shape().to_vec();
        Ok(Self {
            inner: Rc::new(RefCell::new(InnerTensor {
                op: Operation::Add,
                value: result,
                grad: ArrayD::zeros(result_shape),
                predecessors: TensorPredecessor::Two {
                    left: left.inner.clone(),
                    right: right.inner.clone(),
                },
            })),
        })
    }

    /// Element-wise multiplication of two Tensors
    fn mult(left: Self, right: Self) -> Result<Self> {
        let result = &left.inner.borrow().value * &right.inner.borrow().value;
        let result_shape: Vec<usize> = result.shape().to_vec();
        Ok(Self {
            inner: Rc::new(RefCell::new(InnerTensor {
                op: Operation::Mult,
                value: result,
                grad: ArrayD::zeros(result_shape),
                predecessors: TensorPredecessor::Two {
                    left: left.inner.clone(),
                    right: right.inner.clone(),
                },
            })),
        })
    }
}

/// Internal Tensor, contains the actual data
/// as well as tracking gradients, predecessors,
/// and the operation which created it
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

// Function for performing backpropagation with InnerTensors
impl InnerTensor {
    /// Calculate the gradient of the Tensor during backpropagation
    fn backwards(&self) -> Result<()> {
        match self.op {
            Operation::Add => match &self.predecessors {
                TensorPredecessor::None => Err(TensorError::PredecessorMismatch {
                    operation: Operation::Add,
                })
                .context("No predecessors found for Add node during backpropagation"),
                TensorPredecessor::One(_) => Err(TensorError::PredecessorMismatch {
                    operation: Operation::Add,
                })
                .context("Only one predecessor found for Add node during backpropagation"),
                TensorPredecessor::Two { left, right } => {
                    left.borrow_mut().grad += &self.grad;
                    right.borrow_mut().grad += &self.grad;
                    Ok(())
                }
            },
            Operation::Concat { axis, size } => {
                todo!()
            }
            Operation::Conv2D { pad, stride } => {
                todo!()
            }
            Operation::Created => {
                todo!()
            }
            Operation::Cosh => {
                todo!()
            }
            Operation::Mult => match &self.predecessors {
                TensorPredecessor::Two { left, right } => {
                    Zip::from(&mut left.borrow_mut().grad)
                        .and(&right.borrow().value)
                        .and(&self.grad)
                        .for_each(|l, &r, &o| *l += r * o);
                    Zip::from(&mut right.borrow_mut().grad)
                        .and(&left.borrow().value)
                        .and(&self.grad)
                        .for_each(|r, &l, &o| *r += l * o);
                    // NOTE: Previous implementation
                    // left.borrow_mut().grad =
                    //     &(right.borrow().value) * &self.grad + &(left.borrow().grad);
                    // right.borrow_mut().grad =
                    //     &(left.borrow().value) * &self.grad + &(right.borrow().grad);
                    Ok(())
                }
                _ => Err(TensorError::PredecessorMismatch {
                    operation: Operation::Add,
                })
                .context(
                    "Incorrect number of predecessors found for Mult node during backpropagation",
                ),
            },
            Operation::MatMult => {
                todo!()
            }
            Operation::Pow(exp) => {
                todo!()
            }
            Operation::Sinh => {
                todo!()
            }
            Operation::Split { axis, index, first } => {
                todo!()
            }
            Operation::Sub => {
                todo!()
            }
            Operation::Sum(axis) => {
                todo!()
            }
            _ => todo!(),
        }
    }
}

// Functions to access information about the
// value help by the inner tensor (functions
// which are basically just passed to the ndarray)
impl InnerTensor {
    // Passthrough functions for the Value
    fn ndim(&self) -> usize {
        self.value.ndim()
    }

    fn len(&self) -> usize {
        self.value.len()
    }

    fn shape(&self) -> &[usize] {
        self.value.shape()
    }

    fn is_empty(&self) -> bool {
        self.value.is_empty()
    }

    // Passthrough functions for the grad

    fn grad_ndim(&self) -> usize {
        self.grad.ndim()
    }

    fn grad_len(&self) -> usize {
        self.grad.len()
    }

    fn grad_shape(&self) -> &[usize] {
        self.grad.shape()
    }

    fn grad_is_empty(&self) -> bool {
        self.grad.is_empty()
    }
}

// Functions for modifying the inner Tensor
impl InnerTensor {
    /// Set the values of grad to zero
    fn zero_grad(&mut self) {
        self.grad.fill(0f64);
    }
}

// Functions for creating InnerTensors
impl InnerTensor {
    /// Create a new InnerTensor object with `shape`, filled with zeros
    fn new_zeros(shape: &[usize]) -> Result<Self> {
        // This shouldn't fail, but for consistancy returning a result
        Ok(Self {
            op: Operation::Created,
            value: ArrayD::zeros(ndarray::IxDyn(shape)),
            grad: ArrayD::zeros(ndarray::IxDyn(shape)),
            predecessors: TensorPredecessor::None,
        })
    }

    /// Create a new InnerTensor object with `shape`, filled with ones
    fn new_ones(shape: &[usize]) -> Result<Self> {
        Ok(Self {
            op: Operation::Created,
            value: ArrayD::ones(ndarray::IxDyn(shape)),
            grad: ArrayD::zeros(ndarray::IxDyn(shape)),
            predecessors: TensorPredecessor::None,
        })
    }

    /// Create a new InnerTensor object with `shape`, filled with `elem`
    fn new_from_elem(shape: &[usize], elem: f64) -> Result<Self> {
        Ok(Self {
            op: Operation::Created,
            value: ArrayD::from_elem(ndarray::IxDyn(shape), elem),
            grad: ArrayD::zeros(ndarray::IxDyn(shape)),
            predecessors: TensorPredecessor::None,
        })
    }
    /// Create a new InnerTensor object with `shape`, filled with
    /// values drawn from a uniform random distribution between 0.0 and 1.0
    fn new_rand(shape: &[usize]) -> Result<Self> {
        Ok(Self {
            op: Operation::Created,
            value: ArrayD::random(ndarray::IxDyn(shape), Uniform::new(0.0, 1.0)),
            grad: ArrayD::zeros(ndarray::IxDyn(shape)),
            predecessors: TensorPredecessor::None,
        })
    }

    /// Create a new InnerTensor object with `shape`, filled with values
    /// drawn from a random normal distribution with mean 0 and std 1
    fn new_rand_norm(shape: &[usize]) -> Result<Self> {
        Ok(Self {
            op: Operation::Created,
            value: ArrayD::random(
                ndarray::IxDyn(shape),
                Normal::new(0f64, 1f64).context("Failed to create a normal distribution")?,
            ),
            grad: ArrayD::zeros(shape),
            predecessors: TensorPredecessor::None,
        })
    }

    /// Create a new InnerTensor object with `shape`, filled with values
    /// drawn from a random normal distribution with mean `mean`, and
    /// standard deviation `stdev`
    fn new_rand_norm_with_param(shape: &[usize], mean: f64, stdev: f64) -> Result<Self> {
        Ok(Self {
            op: Operation::Created,
            value: ArrayD::random(
                ndarray::IxDyn(shape),
                Normal::new(mean, stdev).context("Failed to create normal distribution")?,
            ),
            grad: ArrayD::zeros(shape),
            predecessors: TensorPredecessor::None,
        })
    }

    /// Create a new InnerTensor object with `shape`, filled with
    /// values drawn from the provided distribution
    fn new_rand_with_dist<IdS: Distribution<f64>>(shape: &[usize], dist: IdS) -> Result<Self> {
        Ok(Self {
            op: Operation::Created,
            value: ArrayD::random(shape, dist),
            grad: ArrayD::zeros(shape),
            predecessors: TensorPredecessor::None,
        })
    }
}

/// An enum to describe the possible predecessors of a Tensor
enum TensorPredecessor {
    /// If the Tensor has no predecessors (i.e. was directly created rather than from a calculation)
    None,
    /// If the Tensor was created by a unary operation
    One(Rc<RefCell<InnerTensor>>),
    /// If the Tensor was created by a Binary Operation
    Two {
        left: Rc<RefCell<InnerTensor>>,
        right: Rc<RefCell<InnerTensor>>,
    },
}

/// Enum describing the operation that created a tensor
#[derive(Debug, Clone)]
enum Operation {
    // Creation
    /// Tensor created
    Created,

    // Unary Operations
    /// Hyperbolic Cosine
    Cosh,
    /// Cosine Function
    Cos,
    /// Exponentiation base e
    Exp,
    /// Exponentiation from any base
    ExpBase(f64),
    /// Heaviside step function
    Heaviside(f64),
    /// Raised to an exponent
    Pow(f64),
    /// Rectified Linear Unit
    Relu,
    /// Sine function
    Sin,
    /// Hyperbolic Sine
    Sinh,
    /// Split a Tensor along `axis`, at position `size`
    Split {
        axis: Axis,
        index: usize,
        first: bool,
    },
    /// Sum along specified axis
    Sum(Axis),
    /// Tangent function
    Tan,
    /// Hyperbolic Tangent function
    Tanh,

    // Binary Operations
    /// Element-wise addition
    Add,
    /// Concatenate two Tensors along specified axis
    Concat { axis: Axis, size: isize },
    /// 2D Convolution
    Conv2D { pad: Option<f64>, stride: isize },
    /// Element-wise division
    Div,
    /// Element-wise multiplication
    Mult,
    /// Matrix multiplication of two Tensors
    MatMult,
    /// Element wise subtraction
    Sub,
}

// SECTION: Errors
#[derive(thiserror::Error, Debug)]
pub enum TensorError {
    #[error("Dimensions do not match for operation: {operation:?}")]
    DimensionMismatch { operation: Operation },
    #[error("Incorrect number of predecessor Tensors for operation: {operation:?}")]
    PredecessorMismatch { operation: Operation },
}

#[cfg(test)]
mod test_creation {
    use super::*;

    #[test]
    fn test_create_zeros() -> Result<()> {
        // Test creating an InnerTensor filled with Zeros
        let test_inner_tensor = InnerTensor::new_zeros(&[4, 5, 1])?;
        assert_eq!(test_inner_tensor.ndim(), 3);
        assert_eq!(test_inner_tensor.len(), 20);
        assert_eq!(test_inner_tensor.shape(), [4, 5, 1]);
        assert_eq!(test_inner_tensor.is_empty(), false);
        // Test creating a Tensor filled with Zeros
        let test_tensor = Tensor::new_zeros(&[4, 4, 1])?;
        assert_eq!(test_tensor.ndim(), 3);
        assert_eq!(test_tensor.len(), 16);
        assert_eq!(test_tensor.shape(), [4, 4, 1]);
        assert_eq!(test_tensor.is_empty(), false);

        Ok(())
    }

    #[test]
    fn test_create_ones() -> Result<()> {
        // Test creating an InnerTensor filled with Zeros
        let test_inner_tensor = InnerTensor::new_ones(&[4, 5, 1])?;
        assert_eq!(test_inner_tensor.ndim(), 3);
        assert_eq!(test_inner_tensor.len(), 20);
        assert_eq!(test_inner_tensor.shape(), [4, 5, 1]);
        assert_eq!(test_inner_tensor.is_empty(), false);

        // Test creating a Tensor filled with Zeros
        let test_tensor = Tensor::new_ones(&[4, 4, 1])?;
        assert_eq!(test_tensor.ndim(), 3);
        assert_eq!(test_tensor.len(), 16);
        assert_eq!(test_tensor.shape(), [4, 4, 1]);
        assert_eq!(test_tensor.is_empty(), false);

        Ok(())
    }

    #[test]
    fn test_create_rand() -> Result<()> {
        // Test creating an InnerTensor filled with random values
        let test_inner_tensor = InnerTensor::new_rand(&[5, 4, 5])?;
        assert_eq!(test_inner_tensor.ndim(), 3);
        assert_eq!(test_inner_tensor.len(), 100);
        assert_eq!(test_inner_tensor.shape(), [5, 4, 5]);
        assert_eq!(test_inner_tensor.is_empty(), false);

        // Test creating a Tensor filled with random values
        let test_tensor = Tensor::new_rand(&[5, 4, 5])?;
        assert_eq!(test_tensor.ndim(), 3);
        assert_eq!(test_tensor.len(), 100);
        assert_eq!(test_tensor.shape(), [5, 4, 5]);
        assert_eq!(test_tensor.is_empty(), false);

        Ok(())
    }
}
