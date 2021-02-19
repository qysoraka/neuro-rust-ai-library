use arrayfire::*;

use crate::tensor::*;

pub struct BatchIterator<'a> {
    data: (&'a Tensor, &'a Tensor),
    num_samples: u64,
    batch_size: u64,
    batch: u64,
    num_batches: u64
}

impl<'a> BatchIterator<'a> {

    /// Creates a batch iterator of given size for the two Tensors.
    ///
    /// # Arguments
    /// * `data` - tuple of reference to the Tensors.
    /// * `batch_size` - size of the mini-batches
    ///
    pub fn new(data: (&'a Tensor, &'a Tensor), batch_size: u64) -> BatchIterator<'a> {
        // Check that both tensors have the same number of samples
        assert_eq!(data.0.dims().get()[3], data.1.dims().get()[3]);
        let num_samples = data.0.dims().get()[3];

        let (batch_size, num_batches) = if batch_size < num_samples {
            let num_batches = (num_samples as f64 / batch_size as f64).ceil() as u64;
            (batch_size, num_batches)
        } else {
            (num_samples, 1)
        };

        BatchIterator {
            data