use neuro::activations::Activation;
use neuro::data::ImageDataSetBuilder;
use neuro::errors::*;
use neuro::layers::{Dense, Conv2D, Padding, MaxPool2D, Dropout, Flatten};
use neuro::losses;
use neuro::metrics::Metrics;
use neuro::models::Network;
use neuro::optimizers::{RMSProp, AdaDelta, Adam};
use neuro::tensor::*;

use std::path::Path;
use neuro::initializers::Initializer;

fn main() -> Result<(), Error> {

    // Load the data
    let path = Path::new("datasets/cifar10");
    let data = ImageDataSetBuilder::from_dir(&path, (32, 32))
        .valid_split(0.1)
        .one_hot_encode()
        .scale(1./255.)
        .build()?;
    println!("{}", data);

    // Create the neural network
    let mut nn = Network::new(Dim::new(&[32, 32, 3, 1]), losses::SoftmaxCrossEntropy::new(), Adam::new(0.001), None)?;
    nn.add(Conv2D::new(32, (3, 3), (1, 1), Padding::Same));
    nn.add(Conv2D::new(32, (3, 3), (1, 1), Padding::Same));
    nn.add(