//! Metrics used to assess the performance of the neural network.
use arrayfire::*;

use crate::tensor::*;

/// Declaration of the metrics.
///
/// Only the accuracy is currently implemented.
#[derive(Debug)]
pub enum Metrics {
    Accuracy,
    /*
    FScore,
    LogLoss,
    MeanAbsoluteError,
    MeanSquaredError,
    RSquared,
    */
}

impl Metrics {
    pub(crate) fn eval(&self, y_pred: &Tensor, y_true: &Tensor) -> PrimitiveType {
        match self {
            Metrics::Accuracy => {
                let batch_size = y_true.dims().get()[3];
                let num_classes = y_true.dims().get()[0];


                let (predicted_class, true_class) = if num_classes == 1 {
                    let predicted_class = select(&constant(1u32, y_pred.dims()), &ge(y_pred, &0.5, true), &constant(0u32, y_pred.dims()));
                    let true_class = select(&constant(1u32, y_true.dims()), &ge(y_true, &0.5, true), &constant(0u32, y_true.dims()))