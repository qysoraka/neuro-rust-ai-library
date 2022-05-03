//! Dropout layer
use arrayfire::*;
use rand::prelude::*;
use std::fmt;

use crate::errors::Error;
use crate::io::{write_scalar, read_scalar};
use crate::layers: