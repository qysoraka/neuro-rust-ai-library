//! Collection of layers used to create neural networks.
use arrayfire::*;

use crate::errors::Error;
use crate::regularizers::*;
use crate::tensor::*;

// Public re-exports
pub use self::batch_normalization::BatchNorm;
pub use self::conv2d::Conv2D;
pub use self::conv2d::Padding;
pub use self::dense::Dense;
pub use self::dropout::Dropout;
pub use self::flatten::Flatten;
pub use self::max_pooling::MaxPool2D;

mod batch_normalization;
mod conv2d;
mod dense;
mod dropout;
mod flatten;
mod max_pooling;


/// Public trait defining the behaviors of a layer.
pub trait Layer: std::fmt::Display {
    /// Returns the name of the layer.
    fn name(&self) -> &str;

    /// Initializes the parameters of the layer.
    fn initialize_parameters(&mut self, input_shape: Dim4);

    /// Computes the activation of the layer during the forward pass.
    fn compute_activation(&self, input: &Tensor) -> Tensor;

    /// Computes the forward pass and stores intermediate values for efficient backpropagation.
    fn compute_activation_mut(&mut self, input: &Tensor) -> Tensor;

    /// Computes the backward pass through the layer.
    fn compute_dactivation_mut(&mut self, input: &Tensor) -> Tensor;

    /// Returns the shape of the output.
    fn output_shape(&self) -> Dim;

    /// Returns the trainable parameters of the layer.
    fn parameters(&self) -> Option<Vec<&Tensor>> { None }

    /// Returns the trainable parameters of the layer and their derivatives.
    fn parameters_mut(&mut self) -> Op