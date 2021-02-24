//! Parameters initialization methods.
use arrayfire::*;
use std::str::FromStr;

use crate::tensor::*;

/// Used to generate the initial values for the parameters of the model.
#[derive(Debug, Copy, Clone)]
pub enum Initializer {
    /// Given constant value.
    Constant(PrimitiveType),
    /// Normal distribution s