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
        let activation: Vec<u8> = group.dataset("activation").and_then(|ds| ds.read_raw::<u8>()).expect("Could not retrieve the activation.");
        let weights = group.dataset("weights").and_then(|ds| ds.read_raw::<H5Tensor>()).expect("Could not retrieve the weights.");
        let biases = group.dataset("biases").and_then(|ds| ds.read_raw::<H5Tensor>()).expect("Could not retrieve the biases.");
        let input_shape = group.dataset("input_shape").and_then(|ds| ds.read_raw::<[u64; 4]>()).expect("Could not retrieve the input shape.");
        let output_shape = group.dataset("output_shape").and_then(|ds| ds.read_raw::<[u64; 4]>()).expect("Could not retrieve the output shape.");
        let regularizer = Regularizer::from_hdf5_group(group);
        let weights_initializer = group.dataset("weights_initializer").and_then(|ds| ds.read_raw::<H5Initializer>()).expect("Could not retrieve the weights initializer.");
        let biases_initializer = group.dataset("biases_initializer").and_then(|ds| ds.read_raw::<H5Initializer>()).expect("Could not retrieve the biases initializer.");

        Box::new(Self {
            units: units[0],
            activation: activation[0].try_into().expect("Could not create activation variant."),
            weights: Tensor::from(&weights[0]),
            dweights: Tensor::new_empty_tensor(),
            biases: Tensor::from(&biases[0]),
            dbiases: Tensor::new_empty_tensor(),
            input_shape: Dim::new(&(input_shape[0])),
            output_shape: Dim::new(&(output_shape[0])),
            linear_activation: None,
            previous_input: None,
            weights_initializer: Initializer::from(&weights_initializer[0]),
            biases_initializer: Initializer::from(&biases_initializer[0]),
            regularizer,
        })
    }
}

impl Layer for Dense
{
    fn name(&self) -> &str {
        Self::NAME
    }

    fn initialize_parameters(&mut self, input_shape: Dim) {
    