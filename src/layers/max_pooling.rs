//! 2D max pooling layer
use arrayfire::*;
use std::fmt;

use crate::errors::Error;
use crate::layers::Layer;
use crate::tensor::*;

/// Defines a 2D max pooling layer.
pub struct MaxPool2D {
    pool_size: (u64, u64),
    stride: (u64, u64),
    input_shape: Dim,
    output_shape: Dim,
    row_indices: Array<i32>,
    col_indices: Array<i32>,
}

impl MaxPool2D {

    pub(crate) const NAME: &'static str = "MaxPool2D";

    /// Creates a 2D max pooling layer.
    ///
    /// By default, the horizontal and vertical strides are set to the height and width of the pooling window.
    ///
    /// # Arguments
    ///
    /// * `pool_size` - The height and width of the pooling window.
    pub fn new(pool_size: (u64, u64)) -> Box<MaxPool2D> {
        Box::new(MaxPool2D {
            pool_size,
            stride: pool_size,
            input_shape: Dim::new(&[0, 0, 0, 0]),
            output_shape: Dim::new(&[0, 0, 0, 0]),
            row_indices: Array::new(&[0], Dim4::new(&[1, 1, 1, 1])),
            col_indices: Array::new(&[0], Dim4::new(&[1, 1, 1, 1])),
 