//! Dense layer
use arrayfire::*;
use std::convert::TryInto;
use std::fmt;

use crate::activations::*;
use crate::errors::Error;
use crate::layers::*;
use crate::initializers::*;
use crate::regularizers::*;
use crate::tensor::*;


/// Defines a dense (or fully connected) layer.
pub struct Dense
{
    units: u64,
    activation: Activation,
    weights: Tensor,
    dweights: Tensor,
    biases: Tensor,
    dbiases: Tensor,
    input_shape: Dim,
    output_shape: Dim,
    linear_activation: Option<Tensor>,
    previous_input: Option<Tensor>,
    weights_initializer: Initializer,
    biases_initializer: Initializer,
    regularizer: Option<Regularizer>,
}



impl Dense
{
    pub(crate) const NAME: &'static str = "Dense";

    /// Creates a dense layer with given number of units and activation function.
    ///
    /// By default, the weights are initialized with a HeUniform initializer