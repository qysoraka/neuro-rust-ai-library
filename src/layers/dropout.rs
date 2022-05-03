//! Dropout layer
use arrayfire::*;
use rand::prelude::*;
use std::fmt;

use crate::errors::Error;
use crate::io::{write_scalar, read_scalar};
use crate::layers::Layer;
use crate::tensor::*;

/// Defines a dropout layer.
pub struct Dropout {
    drop_rate: f64,
    output_shape: Dim,
    grad: Tensor,
    random_engine: RandomEngine,
    scaling_factor: PrimitiveType
}

impl Dropout {

    pub(crate) const NAME: &'static str = "Dropout";

    /// Creates a dropout layer.
    ///
    /// # Arguments
    ///
    /// * `drop_rate` - The probability that a unit will be dropped.
    ///
    /// # Panics
    ///
    /// The method panics if `rate` is smaller than 0 or greater than 1.
    pub fn new(drop_rate: f64) -> Box<Dropout> {

        if drop_rate < 0. || drop_rate > 1. {
            panic!("The drop rate is invalid.");
        }

        let mut rng = rand::thread_rng();
        let seed: u64 = rng.gen();
        let random_engine = RandomEngine::new(RandomEngineType::PHILOX_4X32_10, Some(seed));

        let scaling_factor = 1. / (1. - drop_rat