use hdf5::Group;
use std::fmt;

use crate::errors::Error;
use crate::layers::Layer;
use crate::tensor::*;

pub struct Flatten {
    input_shape: Dim,
    output_shape: Dim,
}

impl Flatten {
    pub(crate) const NAME: &'static str = "Flatten";

    pub fn new() -> Box<Flatten> {
        Box::new(Flatten {
            input_shape: Dim::new(&[0, 0, 0, 0]),
            output_shape: Dim::new(&[0, 0, 0, 0]),
        })
    }

    pub(crate) fn from_hdf5_group(group: &hdf5::Group) -> Box<Flatten> {
        let input_shape = group.dataset("input_shape").and_then(|ds| ds.read_raw::<[u64; 4]>()).expect("Could not retrieve the input shape.");
        let output_shape = group.dat