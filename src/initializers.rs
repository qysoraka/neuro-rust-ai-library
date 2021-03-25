//! Parameters initialization methods.
use arrayfire::*;
use std::str::FromStr;

use crate::tensor::*;

/// Used to generate the initial values for the parameters of the model.
#[derive(Debug, Copy, Clone)]
pub enum Initializer {
    /// Given constant value.
    Constant(PrimitiveType),
    /// Normal distribution scaled using Glorot scale factor.
    GlorotNormal,
    /// Uniform distribution scaled using Glorot scale factor.
    GlorotUniform,
    /// Normal distribution scaled using He scale factor.
    HeNormal,
    /// Uniform distribution scaled using He scale factor.
    HeUniform,
    /// Normal distribution scaled using Lecun scale factor.
    LecunNormal,
    /// Uniform distribution scaled using Lecun scale factor.
    LecunUniform,
    /// Normal distribution with mean 0 and standard deviation 0.01.
    Normal,
    /// Normal distribution with given mean and standard deviation.
    NormalScaled(PrimitiveType, PrimitiveType),
    /// Ones.
    Ones,
    /// Uniform distribution within -0.01 and 0.01.
    Uniform,
    /// Uniform distribution within the given bounds.
    UniformBounded(PrimitiveType, PrimitiveType),
    /// Zeros.
    Zeros,
}

#[derive(hdf5::H5Type, Clone, Debug)]
#[repr(C)]
pub(crate) struct H5Initializer {
    name: hdf5::types::VarLenUnicode,
    values: hdf5::types::VarLenArray<PrimitiveType>,
}

impl From<&H5Initializer> for Initializer {
    fn from(h5_init: &H5Initializer) -> Self {
        match h5_init.name.as_str() {
            "Constant" => Initializer::Constant(h5_init.values[0]),
            "GlorotNormal" => Initializer::GlorotNormal,
            "GlorotUniform" => Initializer::GlorotUniform,
            "HeNormal" => Initializer::HeNormal,
            "HeUniform" => Initializer::HeUniform,
            "LecunNormal" => Initializer::LecunNormal,
            "LecunUniform" => Initializer::LecunUniform,
            "Normal" => Initializer::Normal,
            "NormalScaled" => Initializer::NormalScaled(h5_init.values[0], h5_init.values[1]),
            "Ones" => Initializer::Ones,
            "Uniform" => Initializer::Uniform,
            "UniformBounded" => Initializer::UniformBounded(h5_init.