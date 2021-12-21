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
    /// By default, the weights are initialized with a HeUniform initializer and the biases with a Zeros initializer.
    pub fn new(units: u64, activation: Activation) -> Box<Dense> {
        Box::new(Dense {
            units,
            activation,
            weights: Tensor::new_empty_tensor(),
            dweights: Tensor::new_empty_tensor(),
            biases: Tensor::new_empty_tensor(),
            dbiases: Tensor::new_empty_tensor(),
            input_shape: Dim::new(&[0, 0, 0, 0]),
            output_shape: Dim::new(&[units, 1, 1, 1]),
            linear_activation: None,
            previous_input: None,
            weights_initializer: Initializer::HeNormal,
            biases_initializer: Initializer::Zeros,
            regularizer: None,
        })
    }

    /// Creates a dense layer with the given parameters.
    pub fn with_param(units: u64,
                      activation: Activation,
                      weights_initializer: Initializer,
                      biases_initializer: Initializer
    ) -> Box<Dense> {
        Box::new(Dense {
            units,
            activation,
            weights: Tensor::new_empty_tensor(),
            dweights: Tensor::new_empty_tensor(),
            biases: Tensor::new_empty_tensor(),
            dbiases: Tensor::new_empty_tensor(),
            input_shape: Dim::new(&[0, 0, 0, 0]),
            output_shape: Dim::new(&[units, 1, 1, 1]),
            linear_activation: None,
            previous_input: None,
            weights_initializer,
            biases_initializer,
            regularizer: None,
        })
    }

    pub(crate) fn from_hdf5_group(group: &hdf5::Group) -> Box<Self> {
        let _ = hdf5::silence_errors();
        let units = group.dataset("units").and_then(|ds| ds.read_raw::<u64>()).expect("Could not retrieve the number of units.");
        let activation: Vec<u8> = group.dataset("activation").and_then(|ds| ds.read_raw::<u8>()).expect("Could not ret