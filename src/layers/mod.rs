//! Collection of layers used to create neural networks.
use arrayfire::*;

use crate::errors::Error;
use crate::regularizers::*;
use crate::tensor::*;

// Public re-exports
pub use self::batch_normalization::BatchNorm;
pub use self::conv2d::Conv2D;
pub use self::conv2d::Padding;
pub use self::dense::Dense;
pub use self::dropout::Dropout;
pub use self::flatten::Flatten;
pub use self::max_pooling::MaxPool2D;

mod batch_normalization;
mod conv2d;
mod dense;
mod dropout;
mod flatten;
mod max_pooling;


/// Public trait defining the behaviors of a layer.
pub trait Layer: s