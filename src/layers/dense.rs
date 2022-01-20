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
        let fan_in = input_shape.get()[0] * input_shape.get()[1] * input_shape.get()[2];
        let fan_out = self.units;
        self.weights = self.weights_initializer.new_tensor(Dim::new(&[fan_out, fan_in, 1, 1]), fan_in, fan_out);
        self.biases = self.biases_initializer.new_tensor(Dim::new(&[fan_out, 1, 1, 1]), fan_in, fan_out);
        self.input_shape = input_shape;
    }

    fn compute_activation(&self, input: &Tensor) -> Tensor {
        let linear_activation = add(&matmul(&self.weights, &input, MatProp::NONE, MatProp::NONE), &self.biases, true);
        self.activation.eval(&linear_activation)
    }

    fn compute_activation_mut(&mut self, input: &Tensor) -> Tensor {
        let linear_activation = add(&matmul(&self.weights, input, MatProp::NONE, MatProp::NONE), &self.biases, true);
        let nonlinear_activation = self.activation.eval(&linear_activation);

        // Save input and linear activation for efficient backprop
        self.previous_input = Some(input.clone());
        self.linear_activation = Some(linear_activation);

        // Return the non linear activation
        nonlinear_activation
    }


    fn compute_dactivation_mut(&mut self, input: &Tensor) -> Tensor {
        match &self.linear_activation {
            Some(linear_activation) => {
                let linear_activation_grad = mul(input, &self.activation.grad(linear_activation), true);
                match &mut self.previous_input {
                    Some(previous_input) => {
                        self.dweights = matmul(&linear_activation_grad, previous_input, MatProp::NONE, MatProp::TRANS).reduce(Reduction::MeanBatches);
                        if let Some(regularizer) = self.regularizer { self.dweights += regularizer.grad(&self.weights) }
                        self.dbiases = linear_activation_grad.reduce(Reduction::MeanBatches);
                    },
                    None => panic!("The previous activations have not been computed!"),
                }
                //matmul(&self.weights, &linear_activation_grad, MatProp::TRANS, MatProp::NONE).reshape(Dim4::new(&[self.input_shape[0], self.input_shape[1], self.input_shape[2], input.batch_size()]))
                matmul(&self.weights, &linear_activation_grad, MatProp::TRANS, MatProp::NONE)
            },
            None => panic!("The linear activations z have not been computed!"),
        }
    }

    fn output_shape(&self) -> Dim4 {
        self.output_shape
    }


    fn parameters(&self) -> Option<Vec<&Tensor>> {
        Some(vec![&self.weights, &self.biases])
    }


    fn parameters_mut(&mut self) -> Option<(Vec<&mut Tensor>, Vec<&Tensor>)> {
        Some((vec![&mut self.weights, &mut self.biases], vec![&self.dweights, &self.dbiases]))
    }


    fn save(&self, group: &hdf5::Group, layer_number: usize) -> Result<(), Error> {
        let group_name = layer_number.to_string() + &String::from("_") + Self::NAME;
        let dense = group.create_group(&group_name)?;

        let units = dense.new_dataset::<u64>().create("units", 1)?;
        units.write(&[self.units])?;

        let activation = dense.new_dataset::<Activation>().create("activation", 1)?;
        activation.write(&[self.activation])?;

        let weights = dense.new_dataset::<H5Tensor>().create("weights", 1)?;
        weights.write(&[H5Tensor::from(&self.weights)])?;

        let biases = dense.new_dataset::<H5Tensor>().create("biases", 1)?;
        biases.write(&[H5Tensor::from(&self.biases)])?;

        let input_shape = dense.new_dataset::<[u64; 4]>().create("input_shape", 1)?;
        input_shape.write(&[*self.input_shape.get()])?;

        let output_shape = dense.new_dataset::<[u64; 4]>().create("output_shape", 1)?;
        output_shape.write(&[*self.output_shape.get()])?;

        let weights_initializer = dense.new_dataset::<H5Initializer>().create("weights_initializer", 1)?;
        self.weights_initializer.save(&weights_initializer)?;

        let biases_initializer = dense.new_dataset::<H5Initializer>().create("biases_initializer", 1)?;
        self.biases_initializer.save(&biases_initializer)?;

        Ok(())
    }

    fn set_regularizer(&mut self, regularizer: Option<Regularizer>) {
        self.regularizer = regularizer;
    }

    fn print(&self) {
        println!("Number of parameters: {}", self.weights.elements() + self.biases.elements());
    }
}

impl fmt::Display for Dense {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} \t\t {} \t\t [{}, {}, {}]", Self::N