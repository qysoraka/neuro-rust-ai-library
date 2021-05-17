//! 2D convolution layer
use arrayfire::*;
use std::convert::{TryFrom};
use std::fmt;

use crate::activations::*;
use crate::errors::Error;
use crate::initializers::*;
use crate::regularizers::*;
use crate::tensor::*;
use super::Layer;

/// Defines the type of padding applied to the inputs.
///
/// * Same: a same convolution is such that the dimensions of the output of the convolution is the
/// same as the dimensions of the input, provided a stride of 1.
/// * Valid: a valid convolution is such that the kernel is moved as long as the shift results in a valid convolution operation. No padding is applied.
#[derive(hdf5::H5Type, Debug, Copy, Clone, PartialEq)]
#[repr(u8)]
pub enum Padding {
    Same = 0,
    Valid = 1,
}

impl TryFrom<u8> for Padding {
    type Error = ();

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            x if x == Padding::Same as u8 => Ok(Padding::Same),
            x if x == Padding::Valid as u8 => Ok(Padding::Valid),
            _ => Err(()),
        }
    }
}


/// Defines a 2D convolution layer.
pub struct Conv2D {
    activation: Activation,
    kernel_size: (u64, u64),
    stride: (u64, u64),
    padding: Padding,
    padding_size: (u64, u64, u64, u64), // top, right, bottom, left
    num_filters: u64,
    input_shape: Dim,
    output_shape: Dim,
    weights: Tensor,
    biases: Tensor,
    dweights: Tensor,
    dbiases: Tensor,
    linear_activation: Option<Tensor>,
    previous_activation: Option<Tensor>,
    reshaped_input: Tensor,
    weights_initializer: Initializer,
    biases_initializer: Initializer,
    regularizer: Option<Regularizer>,
}

impl Conv2D {

    pub(crate) const NAME: &'static str = "Conv2D";

    /// Creates a 2D convolution layer with the given parameters.
    ///
    /// By default, a ReLU activation is used and the parameters of the kernels are initialized
    /// using a HeNormal initializer and the biases of the layer a Zeros initializer.
    ///
    /// # Arguments
    ///
    /// * `num_filters` - The number of filters in the layer.
    /// * `kernel_size` - The height and width of the convolution kernels.
    /// * `stride` - The vertical and horizonta