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
/// * Same: a same convolution is such that the dimensions of the output of the convolution is the
/// same as the dimensions of the input, provided a stride of 1.
/// * Valid: a valid convolution is such that the kernel is moved as long as the shift results in a valid convolution operation. No padding is applied.
#[derive(hdf5::H5Type, Debug, Copy, Clone, PartialEq)]
#[repr(u8)]
pub enum Padding {
    Same = 0,
    Valid = 1,
}

impl TryFrom<u8> for Padding {
    type Error = ();

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            x if x == Padding::Same as u8 => Ok(Padding::Same),
            x if x == Padding::Valid as u8 => Ok(Padding::Valid),
            _ => Err(()),
        }
    }
}


/// Defines a 2D convolution layer.
pub struct Conv2D {
    activation: Activation,
    kernel_size: (u64, u64),
    stride: (u64, u64),
    padding: Padding,
    padding_size: (u64, u64, u64, u64), // top, right, bottom, left
    num_filters: u64,
    input_shape: Dim,
    output_shape: Dim,
    weights: Tensor,
    biases: Tensor,
    dweights: Tensor,
    dbiases: Tensor,
    linear_activation: Option<Tensor>,
    previous_activation: Option<Tensor>,
    reshaped_input: Tensor,
    weights_initializer: Initializer,
    biases_initializer: Initializer,
    regularizer: Option<Regularizer>,
}

impl Conv2D {

    pub(crate) const NAME: &'static str = "Conv2D";

    /// Creates a 2D convolution layer with the given parameters.
    ///
    /// By default, a ReLU activation is used and the parameters of the kernels are initialized
    /// using a HeNormal initializer and the biases of the layer a Zeros initializer.
    ///
    /// # Arguments
    ///
    /// * `num_filters` - The number of filters in the layer.
    /// * `kernel_size` - The height and width of the convolution kernels.
    /// * `stride` - The vertical and horizontal stride used for the convolution.
    /// * `padding` - The padding used for the convolution. Must be a variant of Padding.
    pub fn new(num_filters: u64,
               kernel_size: (u64, u64),
               stride: (u64, u64),
               padding: Padding
    ) -> Box<Conv2D> {
        Box::new(Conv2D {
            activation: Activation::ReLU,
            kernel_size,
            stride,
            padding,
            padding_size: (0, 0, 0, 0),
            num_filters,
            input_shape: Dim::new(&[0, 0, 0, 0]),
            output_shape: Dim::new(&[0, 0, 0, 0]),
            weights: Tensor::new_empty_tensor(),
            biases: Tensor::new_empty_tensor(),
            dweights: Tensor::new_empty_tensor(),
            dbiases: Tensor::new_empty_tensor(),
            linear_activation: None,
            previous_activation: None,
            reshaped_input: Tensor::new_empty_tensor(),
            weights_initializer: Initializer::HeNormal,
            biases_initializer: Initializer::Zeros,
            regularizer: None,
        })
    }

    /// Creates a 2D convolution layer with the given parameters.
    ///
    /// By default, the parameters of the kernels are initialized using a HeUniform initializer and the biases
    /// of the layer a Zeros initializer.
    ///
    /// # Arguments
    ///
    /// * `num_filters` - The number of filters in the layer.
    /// * `kernel_size` - The height and width of the convolution kernels.
    /// * `stride` - The vertical and horizontal stride used for the convolution.
    /// * `padding` - The padding used for the convolution. Must be a variant of Padding.
    /// * `activation` - The activation function used by the layer.
    /// * `weights_initializer` - The initializer used to initialize the weights of the layer.
    /// * `biases_initializer` - The initializer used to initialize the biases of the layer.
    pub fn with_param(num_filters: u64,
                      kernel_size: (u64, u64),
                      stride: (u64, u64),
                      padding: Padding,
                      activation: Activation,
                      weights_initializer: Initializer,
                      biases_initializer: Initializer
    ) -> Box<Conv2D> {

        Box::new(Conv2D {
            activation,
            kernel_size,
            stride,
            padding,
            padding_size: (0, 0, 0, 0),
            num_filters,
            input_shape: Dim::new(&[0, 0, 0, 0]),
            output_shape: Dim::new(&[0, 0, 0, 0]),
            weights: Tensor::new_empty_tensor(),
            biases: Tensor::new_empty_tensor(),
            dweights: Tensor::new_empty_tensor(),
            dbiases: Tensor::new_empty_tensor(),
            linear_activation: None,
            previous_activation: None,
            reshaped_input: Tensor::new_empty_tensor(),
            weights_initializer,
            biases_initializer,
            regularizer: None,
        })
    }

    pub(crate) fn from_hdf5_group(group: &hdf5::Group) -> Box<Conv2D> {
        let activation = group.dataset("activation").and_then(|ds| ds.read_raw::<Activation>()).expect("Could not retrieve the activation function.");
        let kernel_size = group.dataset("kernel_size").and_then(|ds| ds.read_raw::<[u64; 2]>()).expect("Could not retrieve the kernel size.");
        let stride = group.dataset("stride").and_then(|ds| ds.read_raw::<[u64; 2]>()).expect("Could not retrieve the stride.");
        let padding = group.dataset("padding").and_then(|ds| ds.read_raw::<Padding>()).expect("Could not retrieve the padding.");
        let padding_size = group.dataset("padding_size").and_then(|ds| ds.read_raw::<[u64; 4]>()).expect("Could not retrieve the pading size.");
        let num_filters = group.dataset("num_filters").and_then(|ds| ds.read_raw::<u64>()).expect("Could not retrieve the number of filters.");
        let input_shape = group.dataset("input_shape").and_then(|value| value.read_raw::<[u64; 4]>()).expect("Could not retrieve the input shape.");
        let output_shape = group.dataset("output_shape").and_then(|value| value.read_raw::<[u64; 4]>()).expect("Could not retrieve the output shape.");
        let weights = group.dataset("weights").and_then(|ds| ds.read_raw::<H5Tensor>()).expect("Could not retrieve the weights.");
        let biases = group.dataset("biases").and_then(|ds| ds.read_raw::<H5Tensor>()).expect("Could not retrieve the biases.");
        let weights_initializer = group.dataset("weights_initializer").and_then(|ds| ds.read_raw::<H5Initializer>()).expect("Could not retrieve the weights initializer.");
        let biases_initializer = group.dataset("biases_initializer").and_then(|ds| ds.read_raw::<H5Initializer>()).expect("Could not retrieve the biases initializer.");
        let regularizer = Regularizer::from_hdf5_group(group);

        Box::new(Conv2D {
            activation: activation[0],
            kernel_size: (kernel_size[0][0], kernel_size[0][1]),
            stride: (stride[0][0], stride[0][1]),
            padding: padding[0],
            padding_size: (padding_size[0][0], padding_size[0][1], padding_size[0][2], padding_size[0][3]),
            num_filters: num_filters[0],
            input_shape: Dim::new(&input_shape[0]),
            output_shape: Dim::new(&output_shape[0]),
            weights: Tensor::from(&weights[0]),
            biases: Tensor::from(&biases[0]),
            dweights: Tensor::new_empty_tensor(),
            dbiases: Tensor::new_empty_tensor(),
            linear_activation: None,
            previous_activation: None,
            reshaped_input: Tensor::new_empty_tensor(),
            weights_initializer: Initializer::from(&weights_initializer[0]),
            biases_initializer: Initializer::from(&biases_initializer[0]),
            regularizer,
        })
    }

    /// Computes the convolution.
    fn compute_convolution(&self, input: &Tensor) -> (Tensor, Tensor) {
        let batch_size = input.dims().get()[3];

        let h_out = self.output_shape.get()[0];
        let w_out = self.output_shape.get()[1];

        // Pad input if necessary
        let padded = self.pad_input(&input);

        // Transform input into column array
        let input_values = match &padded {
            Some(p) => self.img_to_col(&p),
            None => self.img_to_col(input)
        };

        // Compute the convolution and add biases
        let mut conv = add(&matmul(&self.weights, &input_values, MatProp::NONE, MatProp::NONE), &self.biases, true);

        // Reshape to have each mini-batch on the last dimension
        conv = moddims(&conv, Dim4::new(&[self.num_filters, h_out * w_out, 1, batch_size]));

        // Reshape to have correct output dimensions
        let linear_activation = moddims(&transpose(&conv, false), Dim4::new(&[h_out, w_out, self.num_filters, batch_size]));
        (linear_activation, input_values)
    }

    /// Computes the padding that must be added to the images.
    fn compute_padding_size(&mut self, height: u64, width: u64, h_out: u64, w_out: u64) {
        match self.padding {
            Padding::Same => {
                let pad_along_h = std::cmp::max((h_out - 1) * self.stride.0 + self.kernel_size.0 - height, 0);
                let pad_along_w = std::cmp::max((w_out - 1) * self.stride.1 + self.kernel_size.1 - width, 0);
                if pad_along_h != 0 {
                    if pad_along_h % 2 == 0 {
                        self.padding_size.0 = pad_along_h / 2;
                        self.padding_size.2 = pad_along_h / 2;
                    } else {
                        self.padding_size.0 = (pad_along_h - 1) / 2;
                        self.padding_size.2 = (pad_along_h + 1) / 2;
                    }
                }
                if pad_along_w != 0 {
                    if pad_along_w % 2 == 0 {
                        self.padding_size.1 = pad_along_w / 2;
                        self.padding_size.3 = pad_along_w / 2;
                    } else {
                        self.padding_size.1 = (pad_along_w + 1) / 2;
                        self.padding_size.3 = (pad_along_w - 1) / 2;
                    }
                }
            },
            Padding::Valid => {}
        }
    }

    /// Applies the padding to the layer's inputs.
    fn pad_input(&self, input: &Tensor) -> Option<Tensor> {
        let height = input.dims().get()[0];
        let width = input.dims().get()[1];
        let num_channels = input.dims().get()[2];
        let mb_size = input.dims().get()[3];

        // Create padded input
        match self.padding {
            Padding::Same => {
                let pad_top = constant(0.0 as PrimitiveType, Dim4::new(&[self.padding_size.0, width, num_channels, mb_size]));
                let pad_right = constant(0.0 as PrimitiveType, Dim4::new(&[height + self.padding_size.0, self.padding_size.1, num_channels, mb_size]));
                let pad_bottom = constant(0.0 as PrimitiveType, Dim4::new(&[self.padding_size.2, width + self.padding_size.1, num_channels, mb_size]));
                let pad_left = constant(0.0 as PrimitiveType, Dim4::new(&[height + self.padding_size.0 + self.padding_size.2, self.padding_size.3, num_channels, mb_size]));
                let mut padded = join(0, &pad_top, input);
                padded = join(1, &padded, &pad_right);
                padded = join(0, &padded, &pad_bottom);
                padded = join(1, &pad_left, &padded);
                Some(padded)
            },
            Padding::Valid => {
                None
            }
        }
    }

    /// Converts the image into a columns representation.
    ///
    /// This is done for computation speed but there is a memory cost.
    fn img_to_col(&self, input: &Tensor) -> Tensor {
        let num_channels = input.dims().get()[2];
        let mut col = unwrap(input, self.kernel_size.0 as i64, self.kernel_size.1 as i64, self.stride.0 as i64, self.stride.1 as i64, 0, 0, true);
        //col = reorder(&col, Dim4::new(&[0, 2, 1, 3]));
        col = reorder_v2(&col, 0, 2, Some(vec![1, 3]));
        moddims(&col, Dim4::new(&[col.dims().get()[0] * num_channels, col.elements() as u64/(col.dims().get()[0] * num_channels), 1, 1]))
    }

    /// Transforms a columns representation of an image into an image with dimensions height x width x channels.
    fn col_to_img(&self, input: &Tensor) -> Tensor {
        let num_channels = self.input_shape.get()[2];
        let h_out = self.output_shape.get()[0];
        let w_out = self.output_shape.get()[1];
        let num_cols = h_out * w_out;
        let batch_size = input.dims().get()[1] / num_cols;
        let height_padded = (h_out - 1) * self.stride.0 + self.kernel_size.0;
        let width_padded = (w_out - 1) * self.stride.1 + self.kernel_size.1;

        let mut img = moddims(&input, Dim4::new(&[input.dims().get()[0], h_out*w_out, 1, batch_size]));
        //img = reorder(&img, Dim4::new(&[1, 0, 2, 3]));
        img = reorder_v2(&img, 1, 0, Some(vec![2, 3]));
        img = moddims(&img, Dim4::new(&[img.dims().get()[0], self.kernel_size.0 * self.kernel_size.1, num_channel