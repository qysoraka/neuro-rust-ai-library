use hdf5_sys::h5::{H5_INDEX_CRT_ORDER, H5_ITER_INC};
use hdf5_sys::h5p::{H5P_DEFAULT, H5P_CRT_ORDER_INDEXED, H5P_CRT_ORDER_TRACKED, H5P_CLS_LINK_CREATE, H5P_CLS_GROUP_CREATE};
use std::ffi::{CStr, CString};

use crate::tensor::*;

/// Creates an H5 group with creation order tracked and indexed.
///
/// # Arguments
///
/// * `file` - The file in which  the group is created.
/// * `group_name` - Th