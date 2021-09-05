use super::Vec3;
use super::Operations;

use rustacuda::prelude::*;
use rustacuda::error::CudaResult;
use rustacuda::memory::DeviceBuffer;

use std::rc::Rc;

pub struct CudaVec3 {
    buffer: DeviceBuffer<Vec3>,
    stream: Rc<Stream>,
    ops: Rc<Operations>,
}

pub struct CudaFloat {
    buffer: DeviceBuffer<f32>,
    stream: Rc<Stream>,
    ops: Rc<Operations>, 
}

pub struct CudaBool {
    buffer: DeviceBuffer<f32>,
    stream: Rc<Stream>,
    ops: Rc<Operations>,
}

impl CudaVec3 {
    pub fn new(count: usize, stream: &Rc<Stream>, ops: &Rc<Operations>) -> CudaResult<Self> {
        unsafe {
            let buffer = DeviceBuffer::zeroed(count)?;
            Ok(Self {
                buffer,
                stream: Rc::clone(stream),
                ops: Rc::clone(ops),
            })
        }
    }

    pub fn from_buffer(buffer: DeviceBuffer<Vec3>, stream: &Rc<Stream>, ops: &Rc<Operations>) -> Self {
        Self {
            buffer,
            stream: Rc::clone(stream),
            ops: Rc::clone(ops),
        }
    }

    pub fn from_vec(vec: Vec<Vec3>, stream: &Rc<Stream>, ops: &Rc<Operations>) -> CudaResult<Self> {
        unsafe {
            let buffer = DeviceBuffer::from_slice_async(&vec[..], stream.as_ref())?;
            Ok(Self {
                buffer,
                stream: Rc::clone(stream),
                ops: Rc::clone(ops),
            })
        }
    }

    pub fn from_slice(slice: &[Vec3], stream: &Rc<Stream>, ops: &Rc<Operations>) -> CudaResult<Self> {
        unsafe {
            let buffer = DeviceBuffer::from_slice_async(slice, stream.as_ref())?;
            Ok(Self {
                buffer,
                stream: Rc::clone(stream),
                ops: Rc::clone(ops),
            })
        }
    }

    pub fn to_vec(&self) -> CudaResult<Vec<Vec3>> {
        let mut vec: Vec<Vec3> = vec![Vec3::zero(); self.len()];
        self.stream.synchronize()?;
        self.buffer.copy_to(&mut vec[..])?;
        Ok(vec)
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    fn check_can_operate(&self, other: &Self) {
        assert!(Rc::ptr_eq(&self.stream, &other.stream));
        assert!(Rc::ptr_eq(&self.ops, &other.ops));
        assert!(self.len() == other.len());
    }

    fn check_can_operate_with_float(&self, other: &CudaFloat) {
        assert!(Rc::ptr_eq(&self.stream, &other.stream));
        assert!(Rc::ptr_eq(&self.ops, &other.ops));
        assert!(self.len() == other.len());
    }

    fn check_can_operate_with_bool(&self, other: &CudaBool) {
        assert!(Rc::ptr_eq(&self.stream, &other.stream));
        assert!(Rc::ptr_eq(&self.ops, &other.ops));
        assert!(self.len() == other.len());
    }

    pub fn add(&mut self, rhs: &mut Self) -> CudaResult<Self> {
        self.check_can_operate(rhs);
        let buffer = self.ops.vec3_add(&mut self.buffer, &mut rhs.buffer, &self.stream)?;
        Ok(Self::from_buffer(buffer, &self.stream, &self.ops))
    }

    pub fn sub(&mut self, rhs: &mut Self) -> CudaResult<Self> {
        self.check_can_operate(rhs);
        let buffer = self.ops.vec3_sub(&mut self.buffer, &mut rhs.buffer, &self.stream)?;
        Ok(Self::from_buffer(buffer, &self.stream, &self.ops))
    }

    pub fn mul_scalar(&mut self, rhs: &mut CudaFloat) -> CudaResult<Self> {
        self.check_can_operate_with_float(rhs);
        let buffer = self.ops.vec3_mul_scalar(&mut self.buffer, &mut rhs.buffer, &self.stream)?;
        Ok(Self::from_buffer(buffer, &self.stream, &self.ops))
    }

    pub fn div_scalar(&mut self, rhs: &mut CudaFloat) -> CudaResult<Self> {
        self.check_can_operate_with_float(rhs);
        let buffer = self.ops.vec3_div_scalar(&mut self.buffer, &mut rhs.buffer, &self.stream)?;
        Ok(Self::from_buffer(buffer, &self.stream, &self.ops))
    }

    pub fn vec_len(&mut self) -> CudaResult<CudaFloat> {
        let buffer = self.ops.vec3_len(&mut self.buffer, self.stream.as_ref())?;
        Ok(CudaFloat::from_buffer(buffer, &self.stream, &self.ops))
    }

    pub fn vec_len_squared(&mut self) -> CudaResult<CudaFloat> {
        let buffer = self.ops.vec3_len_squared(&mut self.buffer, self.stream.as_ref())?;
        Ok(CudaFloat::from_buffer(buffer, &self.stream, &self.ops))
    }

    pub fn normalized(&mut self) -> CudaResult<Self> {
        let buffer = self.ops.vec3_normalize(&mut self.buffer, self.stream.as_ref())?;
        Ok(Self::from_buffer(buffer, &self.stream, &self.ops))
    }

    pub fn inv(&mut self) -> CudaResult<Self> {
        let buffer = self.ops.vec3_inv(&mut self.buffer, self.stream.as_ref())?;
        Ok(Self::from_buffer(buffer, &self.stream, &self.ops))
    }

    pub fn x(&mut self) -> CudaResult<CudaFloat> {
        let buffer = self.ops.vec3_get_x(&mut self.buffer, self.stream.as_ref())?;
        Ok(CudaFloat::from_buffer(buffer, &self.stream, &self.ops))
    }

    pub fn y(&mut self) -> CudaResult<CudaFloat> {
        let buffer = self.ops.vec3_get_y(&mut self.buffer, self.stream.as_ref())?;
        Ok(CudaFloat::from_buffer(buffer, &self.stream, &self.ops))
    }

    pub fn z(&mut self) -> CudaResult<CudaFloat> {
        let buffer = self.ops.vec3_get_z(&mut self.buffer, self.stream.as_ref())?;
        Ok(CudaFloat::from_buffer(buffer, &self.stream, &self.ops))
    }

    pub fn dot(x: &mut Self, y: &mut Self) -> CudaResult<CudaFloat> {
        x.check_can_operate(y);
        let buffer = x.ops.vec3_dot(&mut x.buffer, &mut y.buffer, x.stream.as_ref())?;
        Ok(CudaFloat::from_buffer(buffer, &x.stream, &x.ops))
    }

    pub fn select(flag: &mut CudaBool, when_true: &mut Self, when_false: &mut Self) -> CudaResult<Self> {
        when_true.check_can_operate(when_false);
        when_true.check_can_operate_with_bool(flag);
        let buffer = flag.ops.vec3_select(&mut flag.buffer, &mut when_true.buffer, &mut when_false.buffer, flag.stream.as_ref())?;
        Ok(Self::from_buffer(buffer, &flag.stream, &flag.ops))
    }
}

impl CudaFloat {
    pub fn new(count: usize, stream: &Rc<Stream>, ops: &Rc<Operations>) -> CudaResult<Self> {
        unsafe {
            let buffer = DeviceBuffer::zeroed(count)?;
            Ok(Self {
                buffer,
                stream: Rc::clone(stream),
                ops: Rc::clone(ops),
            })
        }
    }

    pub fn from_buffer(buffer: DeviceBuffer<f32>, stream: &Rc<Stream>, ops: &Rc<Operations>) -> Self {
        Self {
            buffer,
            stream: Rc::clone(stream),
            ops: Rc::clone(ops),
        }
    }

    pub fn from_vec(vec: Vec<f32>, stream: &Rc<Stream>, ops: &Rc<Operations>) -> CudaResult<Self> {
        unsafe {
            let buffer = DeviceBuffer::from_slice_async(&vec[..], stream.as_ref())?;
            Ok(Self {
                buffer,
                stream: Rc::clone(stream),
                ops: Rc::clone(ops),
            })
        }
    }

    pub fn from_slice(slice: &[f32], stream: &Rc<Stream>, ops: &Rc<Operations>) -> CudaResult<Self> {
        unsafe {
            let buffer = DeviceBuffer::from_slice_async(slice, stream.as_ref())?;
            Ok(Self {
                buffer,
                stream: Rc::clone(stream),
                ops: Rc::clone(ops),
            })
        }
    }

    pub fn as_bool(self) -> CudaBool {  // warning: this not makes actual values to 0 or 1
        CudaBool::from_buffer(self.buffer, &self.stream, &self.ops)
    }

    pub fn to_vec(&self) -> CudaResult<Vec<f32>> {
        let mut vec: Vec<f32> = vec![0.0_f32; self.len()];
        self.stream.synchronize()?;
        self.buffer.copy_to(&mut vec[..])?;
        Ok(vec)
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    fn check_can_operate(&self, other: &Self) {
        assert!(Rc::ptr_eq(&self.stream, &other.stream));
        assert!(Rc::ptr_eq(&self.ops, &other.ops));
        assert!(self.len() == other.len());
    }

    fn check_can_operate_with_vec3(&self, other: &CudaVec3) {
        other.check_can_operate_with_float(self);
    }

    fn check_can_operate_with_bool(&self, other: &CudaBool) {
        assert!(Rc::ptr_eq(&self.stream, &other.stream));
        assert!(Rc::ptr_eq(&self.ops, &other.ops));
        assert!(self.len() == other.len());
    }

    pub fn add(&mut self, rhs: &mut Self) -> CudaResult<Self> {
        self.check_can_operate(rhs);
        let buffer = self.ops.add(&mut self.buffer, &mut rhs.buffer, self.stream.as_ref())?;
        Ok(Self::from_buffer(buffer, &self.stream, &self.ops))
    }

    pub fn sub(&mut self, rhs: &mut Self) -> CudaResult<Self> {
        self.check_can_operate(rhs);
        let buffer = self.ops.sub(&mut self.buffer, &mut rhs.buffer, self.stream.as_ref())?;
        Ok(Self::from_buffer(buffer, &self.stream, &self.ops))
    }

    pub fn mul(&mut self, rhs: &mut Self) -> CudaResult<Self> {
        self.check_can_operate(rhs);
        let buffer = self.ops.mul(&mut self.buffer, &mut rhs.buffer, self.stream.as_ref())?;
        Ok(Self::from_buffer(buffer, &self.stream, &self.ops))
    }

    pub fn div(&mut self, rhs: &mut Self) -> CudaResult<Self> {
        self.check_can_operate(rhs);
        let buffer = self.ops.div(&mut self.buffer, &mut rhs.buffer, self.stream.as_ref())?;
        Ok(Self::from_buffer(buffer, &self.stream, &self.ops))
    }

    pub fn inv(&mut self) -> CudaResult<Self> {
        let buffer = self.ops.inv(&mut self.buffer, self.stream.as_ref())?;
        Ok(Self::from_buffer(buffer, &self.stream, &self.ops))
    }

    pub fn sqrt(&mut self) -> CudaResult<Self> {
        let buffer = self.ops.sqrt(&mut self.buffer, self.stream.as_ref())?;
        Ok(Self::from_buffer(buffer, &self.stream, &self.ops))
    }

    pub fn is_positive(&mut self) -> CudaResult<CudaBool> {
        let buffer = self.ops.is_positive(&mut self.buffer, self.stream.as_ref())?;
        Ok(CudaBool::from_buffer(buffer, &self.stream, &self.ops))
    }

    pub fn is_negative(&mut self) -> CudaResult<CudaBool> {
        let buffer = self.ops.is_negative(&mut self.buffer, self.stream.as_ref())?;
        Ok(CudaBool::from_buffer(buffer, &self.stream, &self.ops))
    }

    pub fn max(x: &mut Self, y: &mut Self) -> CudaResult<Self> {
        x.check_can_operate(y);
        let buffer = x.ops.max(&mut x.buffer, &mut y.buffer, x.stream.as_ref())?;
        Ok(Self::from_buffer(buffer, &x.stream, &x.ops))
    }

    pub fn min(x: &mut Self, y: &mut Self) -> CudaResult<Self> {
        x.check_can_operate(y);
        let buffer = x.ops.min(&mut x.buffer, &mut y.buffer, x.stream.as_ref())?;
        Ok(Self::from_buffer(buffer, &x.stream, &x.ops))
    }

    pub fn pow(x: &mut Self, y: &mut Self) -> CudaResult<Self> {
        x.check_can_operate(y);
        let buffer = x.ops.pow(&mut x.buffer, &mut y.buffer, x.stream.as_ref())?;
        Ok(Self::from_buffer(buffer, &x.stream, &x.ops))
    }

    pub fn compare(x: &mut Self, y: &mut Self) -> CudaResult<CudaBool> {
        x.check_can_operate(y);
        let buffer = x.ops.compare(&mut x.buffer, &mut y.buffer, x.stream.as_ref())?;
        Ok(CudaBool::from_buffer(buffer, &x.stream, &x.ops))
    }

    pub fn select(flag: &mut CudaBool, when_true: &mut Self, when_false: &mut Self) -> CudaResult<Self> {
        when_true.check_can_operate(when_false);
        when_true.check_can_operate_with_bool(flag);
        let buffer = flag.ops.select(&mut flag.buffer, &mut when_true.buffer, &mut when_false.buffer, flag.stream.as_ref())?;
        Ok(Self::from_buffer(buffer, &flag.stream, &flag.ops))
    }
}

impl CudaBool {
    pub fn new_true(count: usize, stream: &Rc<Stream>, ops: &Rc<Operations>) -> CudaResult<Self> {
        let vec = vec![1.0_f32; count];
        unsafe {
            let buffer = DeviceBuffer::from_slice_async(&vec[..], stream.as_ref())?;
            Ok(Self {
                buffer,
                stream: Rc::clone(stream),
                ops: Rc::clone(ops),
            })
        }
    }

    pub fn new_false(count: usize, stream: &Rc<Stream>, ops: &Rc<Operations>) -> CudaResult<Self> {
        unsafe {
            let buffer = DeviceBuffer::zeroed(count)?;
            Ok(Self {
                buffer,
                stream: Rc::clone(stream),
                ops: Rc::clone(ops),
            })
        }
    }

    pub fn from_buffer(buffer: DeviceBuffer<f32>,  stream: &Rc<Stream>, ops: &Rc<Operations>) -> Self {
        Self {
            buffer,
            stream: Rc::clone(stream),
            ops: Rc::clone(ops),
        }
    }

    pub fn from_vec(vec: Vec<f32>, stream: &Rc<Stream>, ops: &Rc<Operations>) -> CudaResult<Self> {
        unsafe {
            let buffer = DeviceBuffer::from_slice_async(&vec[..], stream.as_ref())?;
            Ok(Self {
                buffer,
                stream: Rc::clone(stream),
                ops: Rc::clone(ops),
            })
        }
    }

    pub fn from_slice(slice: &[f32], stream: &Rc<Stream>, ops: &Rc<Operations>) -> CudaResult<Self> {
        unsafe {
            let buffer = DeviceBuffer::from_slice_async(slice, stream.as_ref())?;
            Ok(Self {
                buffer,
                stream: Rc::clone(stream),
                ops: Rc::clone(ops),
            })
        }
    }

    pub fn as_float(self) -> CudaFloat {
        CudaFloat::from_buffer(self.buffer, &self.stream, &self.ops)
    }

    pub fn to_vec(&self) -> CudaResult<Vec<f32>> {
        let mut vec: Vec<f32> = vec![0.0_f32; self.len()];
        self.stream.synchronize()?;
        self.buffer.copy_to(&mut vec[..])?;
        Ok(vec)
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    fn check_can_operate(&self, other: &Self) {
        assert!(Rc::ptr_eq(&self.stream, &other.stream));
        assert!(Rc::ptr_eq(&self.ops, &other.ops));
        assert!(self.len() == other.len());
    }

    fn check_can_operate_with_vec3(&self, other: &CudaVec3) {
        other.check_can_operate_with_bool(self);
    }

    fn check_can_operate_with_float(&self, other: &CudaFloat) {
        other.check_can_operate_with_bool(self);
    }

    pub fn not(&mut self) -> CudaResult<Self> {
        let buffer = self.ops.not(&mut self.buffer, self.stream.as_ref())?;
        Ok(Self::from_buffer(buffer, &self.stream, &self.ops))
    }

    pub fn and(x: &mut Self, y: &mut Self) -> CudaResult<Self> {
        x.check_can_operate(y);
        let buffer = x.ops.and(&mut x.buffer, &mut y.buffer, x.stream.as_ref())?;
        Ok(Self::from_buffer(buffer, &x.stream, &x.ops))
    }

    pub fn or(x: &mut Self, y: &mut Self) -> CudaResult<Self> {
        x.check_can_operate(y);
        let buffer = x.ops.or(&mut x.buffer, &mut y.buffer, x.stream.as_ref())?;
        Ok(Self::from_buffer(buffer, &x.stream, &x.ops))
    }
}