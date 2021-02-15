use neuro::activations::Activation;
use neuro::data::{TabularDataSet, DataSet};
use neuro::errors::*;
use neuro::initializers::*;
use neuro::layers::Dense;
use neuro::losses;
use neuro::metrics;
use neuro::models;
use neuro::optimizers::SGD;
use neuro::tensor::*;

fn main() -> Result<(), Error> {
    // Create the dataset
    let input_values = [0., 0., 0., 1., 1., 0., 1., 1.];
    let x_train = Tensor::new(&input_values, Dim::new(&[2, 1, 1, 4]));
    let output_values = [0., 1., 1., 0.];
    let y_train = Tensor::new(&output_values, Dim::new(&[1, 1, 1, 4]));
    let data = TabularDataSet::from_tensor(x_train.copy(), y_train.copy(), None, None, None, None)?;

    // Create the neural network and a