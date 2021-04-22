use hdf5_sys::h5::{H5_INDEX_CRT_ORDER, H5_ITER_INC};
use hdf5_sys::h5p::{H5P_DEFAULT, H5P_CRT_ORDER_INDEXED, H5P_CRT_ORDER_TRACKED, H5P_CLS_LINK_CREATE, H5P_CLS_GROUP_CREATE};
use std::ffi::{CStr, CString};

use crate::tensor::*;

/// Creates an H5 group with creation order tracked and indexed.
///
/// # Arguments
///
/// * `file` - The file in which  the group is created.
/// * `group_name` - The name of the group.
pub(crate) fn create_group(file: &hdf5::File, group_name: &str) -> hdf5::Group {
    let name = CString::new(group_name).unwrap();
    unsafe {
        let lcpl = hdf5_sys::h5p::H5Pcreate(*H5P_CLS_LINK_CREATE);
        let gcpl = hdf5_sys::h5p::H5Pcreate(*H5P_CLS_GROUP_CREATE);
        hdf5_sys::h5p::H5Pset_link_creation_order(gcpl, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED);
        hdf5_sys::h5g::H5Gcreate2(file.id(), name.as_ptr(), lcpl, gcpl, H5P_DEFAULT);
    }
    file.group(group_name).expect("Could not create the group.")

}

/// Lists the subgroups contained in a group.
///
/// # Return value
///
/// Vector containing the names of the subgroups, listed by creation order.
pub(crate) fn list_subgroups(group: &hdf5::Group) -> Vec<String> {
    extern "C" fn members_callback(
        _id: hdf5_sys::h5i::hid_t, name: *const std::os::raw::c_char, _info: *const hdf5_sys::h5l::H5L_info_t, op_data: 