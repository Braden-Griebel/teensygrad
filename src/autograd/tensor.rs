// Standard Library Use statements
use std::cell::RefCell;
use std::rc::Rc;

// External Crate Uses
use anyhow::{Context, Result};
use ndarray::{ArrayD, Axis};
use ndarray_rand::rand_distr::{Distribution, Normal, Uniform};
use ndarray_rand::RandomExt;

/// A wrapped tensor, which will track computations
/// enabling automatic differentiation
pub struct Tensor {
    inner: Rc<RefCell<InnerTensor>>,
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
    fn backwards(&self) {
        match self.op {
            Operation::Add => {}
            Operation::Concat { axis, size } => {}
            Operation::Conv2D { pad, stride } => {}
            Operation::Created => {}
            Operation::Cosh => {}
            Operation::Mult => {}
            Operation::MatMult => {}
            Operation::Pow(exp) => {}
            Operation::Sinh => {}
            Operation::Split { axis, size } => {}
            Operation::Sub => {}
            Operation::Sum(axis) => {}
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
enum Operation {
    Add,
    Concat { axis: Axis, size: isize },
    Conv2D { pad: Option<f64>, stride: isize },
    Cosh,
    Created,
    Mult,
    MatMult,
    Pow(f64),
    Sinh,
    Split { axis: Axis, size: isize },
    Sub,
    Sum(Axis),
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
