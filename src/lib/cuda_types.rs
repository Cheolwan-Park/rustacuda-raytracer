use super::Vec3;
use super::Operations;

use rustacuda::prelude::*;
use rustacuda::error::CudaResult;
use rustacuda::memory::DeviceBuffer;

use std::rc::Rc;
use std::ops;

pub struct CudaVec3 {
    buffer: DeviceBuffer<Vec3>,
    count: usize,
    stream: Rc<Stream>,
    ops: Rc<Operations>,
}

pub struct CudaFloat {
    buffer: DeviceBuffer<f32>,
    count: usize,
    stream: Rc<Stream>,
    ops: Rc<Operations>, 
}

pub struct CudaBool {
    buffer: DeviceBuffer<f32>,
    count: usize,
    stream: Rc<Stream>,
    ops: Rc<Operations>,
}

impl CudaVec3 {
    pub fn new(count: usize, stream: &Rc<Stream>, ops: &Rc<Operations>) -> CudaResult<Self> {
        let vec = vec![Vec3::zero(); count];
        unsafe {
            let buffer = DeviceBuffer::from_slice_async(&vec[..], stream.as_ref())?;
            Ok(Self {
                buffer,
                count,
                stream: Rc::clone(stream),
                ops: Rc::clone(ops),
            })
        }
    }

    pub fn from_vec(vec: &Vec<Vec3>, stream: &Rc<Stream>, ops: &Rc<Operations>) -> CudaResult<Self> {
        unsafe {
            let buffer = DeviceBuffer::from_slice_async(&vec[..], stream.as_ref())?;
            Ok(Self {
                buffer,
                count: vec.len(),
                stream: Rc::clone(stream),
                ops: Rc::clone(ops),
            })
        }
    }
}

impl CudaFloat {
    pub fn new(count: usize, stream: &Rc<Stream>, ops: &Rc<Operations>) -> CudaResult<Self> {
        let vec = vec![0.0_f32; count];
        unsafe {
            let buffer = DeviceBuffer::from_slice_async(&vec[..], stream.as_ref())?;
            Ok(Self {
                buffer,
                count,
                stream: Rc::clone(stream),
                ops: Rc::clone(ops),
            })
        }
    }

    pub fn from_vec(vec: &Vec<f32>, stream: &Rc<Stream>, ops: &Rc<Operations>) -> CudaResult<Self> {
        unsafe {
            let buffer = DeviceBuffer::from_slice_async(&vec[..], stream.as_ref())?;
            Ok(Self {
                buffer,
                count: vec.len(),
                stream: Rc::clone(stream),
                ops: Rc::clone(ops),
            })
        }
    }
}

impl CudaBool {
    pub fn new_true(count: usize, stream: &Rc<Stream>, ops: &Rc<Operations>) -> CudaResult<Self> {
        let vec = vec![1.0_f32; count];
        unsafe {
            let buffer = DeviceBuffer::from_slice_async(&vec[..], stream.as_ref())?;
            Ok(Self {
                buffer,
                count,
                stream: Rc::clone(stream),
                ops: Rc::clone(ops),
            })
        }
    }

    pub fn new_false(count: usize, stream: &Rc<Stream>, ops: &Rc<Operations>) -> CudaResult<Self> {
        let vec = vec![0.0_f32; count];
        unsafe {
            let buffer = DeviceBuffer::from_slice_async(&vec[..], stream.as_ref())?;
            Ok(Self {
                buffer,
                count,
                stream: Rc::clone(stream),
                ops: Rc::clone(ops),
            })
        }
    }

    pub fn from_vec(vec: &Vec<f32>, stream: &Rc<Stream>, ops: &Rc<Operations>) -> CudaResult<Self> {
        unsafe {
            let buffer = DeviceBuffer::from_slice_async(&vec[..], stream.as_ref())?;
            Ok(Self {
                buffer,
                count: vec.len(),
                stream: Rc::clone(stream),
                ops: Rc::clone(ops),
            })
        }
    }
}

// impl ops::Add<CudaVec3> for CudaVec3 {
//     type Output = CudaVec3;

//     fn add(self, rhs: CudaVec3) -> CudaVec3 {
//         assert!(Rc::ptr_eq(&self.stream, &rhs.stream));
//         assert!(Rc::ptr_eq(&self.ops, &rhs.ops));

//         let buffer = self.ops.vec3_add(&mut self.buffer, &mut rhs.buffer, self.count, self.stream.as_ref()).expect("operation failed");
//         CudaVec3 {
//             buffer,
//             count: self.count,
//             stream: self.stream.clone(),
//             ops: self.ops.clone()
//         }
//     }
// }