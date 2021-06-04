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
    /// * `stride` - The vertical and horizontal stride used for the convolution.
    /// * `padding` - The padding used for the convolution. Must be a variant of Padding.
    pub fn new(num_filters: u64,
               kernel_size: (u64, u64),
               stride: (u64, u64),
               padding: Padding
    ) -> Box<Conv2D> {
        Box::new(Conv2D {
            activation: Activation::ReLU,
            kernel_size,
            stride,
            padding,
            padding_size: (0, 0, 0, 0),
            num_filters,
            input_shape: Dim::new(&[0, 0, 0, 0]),
            output_shape: Dim::new(&[0, 0, 0, 0]),
            weights: Tensor::new_empty_tensor(),
            biases: Tensor::new_empty_tensor(),
            dweights: Tensor::new_empty_tensor(),
            dbiases: Tensor::new_empty_tensor(),
            linear_activation: None,
            previous_activation: None,
            reshaped_input: Tensor::new_empty_tensor(),
            weights_initializer: Initializer::HeNormal,
            biases_initializer: Initializer::Zeros,
            regularizer: None,
        })
    }

    /// Creates a 2D convolution layer with the given parameters.
    ///
    /// By default, the parameters of the kernels are initialized using a HeUniform initializer and the biases
    /// of the layer a Zeros initializer.
    ///
    /// # Arguments
    ///
    /// * `num_filters` - The number of filters in the layer.
    /// * `kernel_size` - The height and width of the convolution kernels.
    /// * `stride` - The vertical and horizontal stride used for the convolution.
    /// * `padding` - The padding used for the convolution. Must be a variant of Padding.
    /// * `activation` - The activation function used by the layer.
    /// * `weights_initializer` - The initializer used to initialize the weights of the layer.
    /// * `biases_initializer` - The initializer used to initialize the biases of the layer.
    pub fn with_param(num_filters: u64,
                      kernel_size: (u64, u64),
                      stride: (u64, u64),
                      padding: Padding,
                      activation: Activation,
                      weights_initializer: Initializer,
                      biases_initializer: Initializer
    ) -> Box<Conv2D> {

        Box::new(Conv2D {
            activation,
            kernel_size,
            stride,
            padding,
            padding_size: (0, 0, 0, 0),
            num_filters,
            input_shape: Dim::new(&[0, 0, 0, 0]),
            output_shape: Dim::new(&[0, 0, 0, 0]),
            weights: Tensor::new_empty_tensor(),
            biases: Tensor::new_empty_tensor(),
            dweights: Tensor::new_empty_tensor(),
            dbiases: Tensor::new_empty_tensor(),
            linear_activation: None,
            previous_activation: None,
            reshaped_input: Tensor::new_empty_tensor(),
            weights_initializer,
            biases_initializer,
            regularizer: None,
        })
    }

    pub(crate) fn from_hdf5_group(group: &hdf5::Group) -> Box<Conv2D> {
        let activation = group.dataset("activation").and_then(|ds| ds.read_raw::<Activation>()).expect("Could not retrieve the activation function.");
        let kernel_size = group.dataset("kernel_size").and_then(|ds| ds.read_raw::<[u64; 2]>()).expect("Could not retrieve the kernel size.");
        let stride = group.dataset("stride").and_then(|ds| ds.read_raw::<[u64; 2]>()).expect("Could not retrieve the stride.");
        let padding = group.dataset("padding").and_then(|ds| ds.read_raw::<Padding>()).expect("Could not retrieve the padding.");
        let padding_size = group.dataset("padding_size").and_then(|ds| ds.read_raw::<[u64; 4]>()).expect("Could not retrieve the pading size.");
        let num_filters = group.dataset("num_filters").and_then(|ds| ds.read_raw::<u64>()).expect("Could not retrieve the number of filters.");
        let input_shape = group.dataset("input_shape").and_then(|value| value.read_raw::<[u64; 4]>()).expect("Could not retrieve the input shape.");
        let output_shape = group.dataset("output_shape").and_then(|value| value.read_raw::<[u64; 4]>()).expect("Could not retrieve the output shape.");
        let weights = group.dataset("weights").and_then(|ds| ds.read_raw::<H5Tensor>()).expect("Could not retrieve the weights.");
        let biases = group.dataset("biases").and_then(|ds| ds.read_raw::<H5Tensor>()).expect("Could not retrieve the biases.");
        let weights_initializer = group.dataset("weights_initializer").and_then(|ds| ds.read_raw::<H5Initializer>(