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
        })
    }


    /// Creates a 2D max pooling layer with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `pool_size` - The height and width of the moving window.
    /// * `stride` - The vertical and horizontal stride.
    pub fn with_param(pool_size: (u64, u64), stride: (u64, u64)) -> Box<MaxPool2D> {
        Box::new(MaxPool2D {
            pool_size,
            stride,
            input_shape: Dim::new(&[0, 0, 0, 0]),
            output_shape: Dim::new(&[0, 0, 0, 0]),
            row_indices: Array::new(&[0], Dim4::new(&[1, 1, 1, 1])),
            col_indices: Array::new(&[0], Dim4::new(&[1, 1, 1, 1])),
        })
    }

    /// Creates a MaxPool2D layer from an HDF5 group.
    pub(crate) fn from_hdf5_group(group: &hdf5::Group) -> Box<MaxPool2D> {
        let pool_size = group.dataset("pool_size").and_then(|ds| ds.read_raw::<[u64; 2]>()).expect("Could not retrieve the pool size.");
        let stride = group.dataset("stride").and_then(|ds| ds.read_raw::<[u64; 2]>()).expect("Could not retrieve the stride.");
        let input_shape = group.dataset("input_shape").and_then(|ds| ds.read_raw::<[u64; 4]>()).expect("Could not retrieve the input shape.");
        let output_shape = group.dataset("output_shape").and_then(|ds| ds.read_raw::<[u64; 4]>()).expect("Could not retrieve the output shape.");

        Box::new(MaxPool2D {
            pool_size: (pool_size[0][0], pool_size[0][1]),
            stride: (stride[0][0], stride[0][1]),
            input_shape: Dim::new(&input_shape[0]),
            output_shape: Dim::new(&output_shape[0]),
            row_indices: Array::new(&[0], Dim4::new(&[1, 1, 1, 1])),
            col_indices: Array::new(&[0], Dim4::new(&[1, 1, 1, 1])),
        })
    }

    /// Computes the maximum value in the pooling window.
    fn max_pool(&self, input: &Tensor) -> (Tensor, Array<i32>, Array<i32>) {
        let cols = unwrap(input, self.pool_size.0 as i64, self.pool_size.1 as i64, self.stride.0 as i64, self.stride.1 as i64, 0, 0, true);
        let cols_reshaped = moddims(&cols, Dim4::new(&[cols.dims().get()[0], cols.elements() as u64 / cols.dims().get()[0], 1, 1]));

        // Computes max values and indices
        let (mut max_values, row_indices_u32) = imax(&cols_reshaped, 0);

        // Creates the output
        let output = moddims(&max_values, Dim4::new(&[self.output_shape.get()[0], self.output_shape.get()[1], input.dims().get()[2], input.dims().get()[3]]));

        // Creates rows and columns indices
        let mut row_indices: Array<i32> = row_indices_u32.cast();
        //row_indices = reorder(&row_indices, Dim4::new(&[1, 0, 2, 3]));
        row_indices = reorder_v2(&row_indices, 1, 0, Some(vec![2, 3]));

        //max_values = reorder(&max_values, Dim4::new(&[1, 0, 2, 3]));
        max_values = reorder_v2(&max_values, 1, 0, Some(vec![2, 3]));
  