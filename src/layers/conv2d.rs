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
/// * Same: a same convolutio