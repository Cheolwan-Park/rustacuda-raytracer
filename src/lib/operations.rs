use rustacuda::prelude::*;
use rustacuda::memory::{DeviceBuffer, DeviceBox};
use rustacuda::error::CudaResult;
use std::ffi::CString;
use std::option::Option;
// use std::marker::Send;

use super::Vec3;

pub struct Operations {
    module: Module,
    context: Context,
    grid_size: u32,
    block_size: u32,
}

impl Operations {
    pub fn new(grid_size: u32, block_size: u32) -> Self {
        rustacuda::init(CudaFlags::empty()).expect("error initializing rustacuda");
    
        let device = Device::get_device(0).expect("error getting device");
        let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device).expect("error creating context");

        let module_data = CString::new(include_str!("../../cuda/operations.ptx")).expect("failed to open .ptx file");
        let module = Module::load_from_string(&module_data).expect("error creating module");
        Self {
            module,
            context,
            grid_size,
            block_size,
        }
    }

    pub fn create_stream(&self) -> Option<Stream> {
        match Stream::new(StreamFlags::NON_BLOCKING, None) {
            Err(e) => {
                println!("error creating stream: {:?}", e);
                None
            },
            Ok(val) => Some(val)
        }
    }

    pub fn slice_to_device<T>(
        slice: &[T], 
        stream: &Stream
    ) -> CudaResult<DeviceBuffer<T>>
    where T: rustacuda_core::DeviceCopy {
        unsafe {
            DeviceBuffer::from_slice_async(slice, stream)
        }
    }

    pub fn to_device<T>(
        obj: &T
    ) -> CudaResult<DeviceBox<T>>
    where T: rustacuda_core::DeviceCopy {
        DeviceBox::new(obj)
    }

    pub fn add(
        &self,
        x: &mut DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.add<<<self.grid_size, self.block_size, 0, stream>>>(
                x.as_device_ptr(),
                y.as_device_ptr(),
                out.as_device_ptr(),
                count
            ))?;
            Ok(out)
        }
    }

    pub fn sub(
        &self,
        x: &mut DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.sub<<<self.grid_size, self.block_size, 0, stream>>>(
                x.as_device_ptr(),
                y.as_device_ptr(),
                out.as_device_ptr(),
                count
            ))?;
            Ok(out)
        }
    }

    pub fn mul(
        &self,
        x: &mut DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.mul<<<self.grid_size, self.block_size, 0, stream>>>(
                x.as_device_ptr(),
                y.as_device_ptr(),
                out.as_device_ptr(),
                count
            ))?;
            Ok(out)
        }
    }

    pub fn div(
        &self,
        x: &mut DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.divide<<<self.grid_size, self.block_size, 0, stream>>>(
                x.as_device_ptr(),
                y.as_device_ptr(),
                out.as_device_ptr(),
                count
            ))?;
            Ok(out)
        }
    }

    pub fn max(
        &self,
        x: &mut DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.float_max<<<self.grid_size, self.block_size, 0, stream>>>(
                x.as_device_ptr(),
                y.as_device_ptr(),
                out.as_device_ptr(),
                count
            ))?;
            Ok(out)
        }
    }

    pub fn min(
        &self,
        x: &mut DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.float_min<<<self.grid_size, self.block_size, 0, stream>>>(
                x.as_device_ptr(),
                y.as_device_ptr(),
                out.as_device_ptr(),
                count
            ))?;
            Ok(out)
        }
    }

    pub fn inv(
        &self,
        x: &mut DeviceBuffer<f32>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.inv<<<self.grid_size, self.block_size, 0, stream>>>(
                x.as_device_ptr(),
                out.as_device_ptr(),
                count
            ))?;
            Ok(out)
        }
    }

    pub fn sqrt(
        &self,
        x: &mut DeviceBuffer<f32>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.float_sqrt<<<self.grid_size, self.block_size, 0, stream>>>(
                x.as_device_ptr(),
                out.as_device_ptr(),
                count
            ))?;
            Ok(out)
        }
    }

    pub fn pow(
        &self,
        x: &mut DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.float_pow<<<self.grid_size, self.block_size, 0, stream>>>(
                x.as_device_ptr(),
                y.as_device_ptr(),
                out.as_device_ptr(),
                count
            ))?;
            Ok(out)
        }
    }

    pub fn is_positive(
        &self,
        x: &mut DeviceBuffer<f32>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.is_positive<<<self.grid_size, self.block_size, 0, stream>>>(
                x.as_device_ptr(),
                out.as_device_ptr(),
                count
            ))?;
            Ok(out)
        }
    }

    pub fn is_negative(
        &self,
        x: &mut DeviceBuffer<f32>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.is_negative<<<self.grid_size, self.block_size, 0, stream>>>(
                x.as_device_ptr(),
                out.as_device_ptr(),
                count
            ))?;
            Ok(out)
        }
    }

    pub fn compare(         // out is true iff x > y
        &self,
        x: &mut DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<f32>> {
        let mut out = self.sub(x, y, count, stream)?;
        self.is_positive(&mut out, count, stream)
    }

    pub fn and(
        &self,
        x: &mut DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.bool_and<<<self.grid_size, self.block_size, 0, stream>>>(
                x.as_device_ptr(),
                y.as_device_ptr(),
                out.as_device_ptr(),
                count
            ))?;
            Ok(out)
        }
    }

    pub fn or(
        &self,
        x: &mut DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.bool_or<<<self.grid_size, self.block_size, 0, stream>>>(
                x.as_device_ptr(),
                y.as_device_ptr(),
                out.as_device_ptr(),
                count
            ))?;
            Ok(out)
        }
    }

    pub fn not(
        &self,
        x: &mut DeviceBuffer<f32>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.bool_not<<<self.grid_size, self.block_size, 0, stream>>>(
                x.as_device_ptr(),
                out.as_device_ptr(),
                count
            ))?;
            Ok(out)
        }
    }

    pub fn select(
        &self,
        flag: &mut DeviceBuffer<f32>,
        when_true: &mut DeviceBuffer<f32>,
        when_false: &mut DeviceBuffer<f32>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.float_select<<<self.grid_size, self.block_size, 0, stream>>>(
                flag.as_device_ptr(),
                when_true.as_device_ptr(),
                when_false.as_device_ptr(),
                out.as_device_ptr(),
                count
            ))?;
            Ok(out)
        }
    }

    pub fn vec3_add(
        &self,
        x: &mut DeviceBuffer<Vec3>, 
        y: &mut DeviceBuffer<Vec3>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<Vec3>> {
        unsafe {
            let out = vec![Vec3::zero(); count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_add<<<self.grid_size, self.block_size, 0, stream>>>(
                x.as_device_ptr(),
                y.as_device_ptr(),
                out.as_device_ptr(),
                count
            ))?;
            Ok(out)
        }
    }

    pub fn vec3_sub(
        &self,
        x: &mut DeviceBuffer<Vec3>, 
        y: &mut DeviceBuffer<Vec3>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<Vec3>> {
        unsafe {
            let out = vec![Vec3::zero(); count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_sub<<<self.grid_size, self.block_size, 0, stream>>>(
                x.as_device_ptr(),
                y.as_device_ptr(),
                out.as_device_ptr(),
                count
            ))?;
            Ok(out)
        }
    }

    pub fn vec3_dot(
        &self,
        x: &mut DeviceBuffer<Vec3>, 
        y: &mut DeviceBuffer<Vec3>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_dot<<<self.grid_size, self.block_size, 0, stream>>>(
                x.as_device_ptr(),
                y.as_device_ptr(),
                out.as_device_ptr(),
                count
            ))?;
            Ok(out)
        }
    }

    pub fn vec3_mul_scalar(
        &self,
        v: &mut DeviceBuffer<Vec3>,
        scalar: &mut DeviceBuffer<f32>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<Vec3>> {
        unsafe {
            let out = vec![Vec3::zero(); count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_mul_scalar<<<self.grid_size, self.block_size, 0, stream>>>(
                v.as_device_ptr(),
                scalar.as_device_ptr(),
                out.as_device_ptr(),
                count
            ))?;
            Ok(out)
        }
    }

    pub fn vec3_div_scalar(
        &self,
        v: &mut DeviceBuffer<Vec3>,
        scalar: &mut DeviceBuffer<f32>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<Vec3>> {
        unsafe {
            let out = vec![Vec3::zero(); count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_div_scalar<<<self.grid_size, self.block_size, 0, stream>>>(
                v.as_device_ptr(),
                scalar.as_device_ptr(),
                out.as_device_ptr(),
                count
            ))?;
            Ok(out)
        }
    }

    pub fn vec3_len(
        &self,
        v: &mut DeviceBuffer<Vec3>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.len<<<self.grid_size, self.block_size, 0, stream>>>(
                v.as_device_ptr(),
                out.as_device_ptr(),
                out.len()
            ))?;
            Ok(out)
        }
    }

    pub fn vec3_len_squared(
        &self,
        v: &mut DeviceBuffer<Vec3>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_len_squared<<<self.grid_size, self.block_size, 0, stream>>>(
                v.as_device_ptr(),
                out.as_device_ptr(),
                out.len()
            ))?;
            Ok(out)
        }
    }

    pub fn vec3_normalize(
        &self,
        v: &mut DeviceBuffer<Vec3>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<Vec3>> {
        unsafe {
            let out = vec![Vec3::zero(); count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_normalize<<<self.grid_size, self.block_size, 0, stream>>>(
                v.as_device_ptr(),
                out.as_device_ptr(),
                out.len()
            ))?;
            Ok(out)
        }
    }

    pub fn vec3_inv(
        &self,
        v: &mut DeviceBuffer<Vec3>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<Vec3>> {
        unsafe {
            let out = vec![Vec3::zero(); count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_inv<<<self.grid_size, self.block_size, 0, stream>>>(
                v.as_device_ptr(),
                out.as_device_ptr(),
                out.len()
            ))?;
            Ok(out)
        }
    }

    pub fn vec3_get_x(
        &self,
        v: &mut DeviceBuffer<Vec3>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_get_x<<<self.grid_size, self.block_size, 0, stream>>>(
                v.as_device_ptr(),
                out.as_device_ptr(),
                out.len()
            ))?;
            Ok(out)
        }
    }

    pub fn vec3_get_y(
        &self,
        v: &mut DeviceBuffer<Vec3>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_get_y<<<self.grid_size, self.block_size, 0, stream>>>(
                v.as_device_ptr(),
                out.as_device_ptr(),
                out.len()
            ))?;
            Ok(out)
        }
    }

    pub fn vec3_get_z(
        &self,
        v: &mut DeviceBuffer<Vec3>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_get_z<<<self.grid_size, self.block_size, 0, stream>>>(
                v.as_device_ptr(),
                out.as_device_ptr(),
                out.len()
            ))?;
            Ok(out)
        }
    }

    pub fn vec3_select(
        &self,
        flag: &mut DeviceBuffer<f32>,
        when_true: &mut DeviceBuffer<Vec3>,
        when_false: &mut DeviceBuffer<Vec3>,
        count: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<Vec3>> {
        unsafe {
            let out = vec![Vec3::zero(); count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_select<<<self.grid_size, self.block_size, 0, stream>>>(
                flag.as_device_ptr(),
                when_true.as_device_ptr(),
                when_false.as_device_ptr(),
                out.as_device_ptr(),
                count
            ))?;
            Ok(out)
        }
    }
}