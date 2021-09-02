use rustacuda::prelude::*;
use rustacuda::memory::{DeviceBuffer, DeviceBox};
use rustacuda::error::CudaResult;
use std::ffi::CString;
use std::option::Option;
use std::sync::{Mutex, Arc};
// use std::marker::Send;

use super::Vec3;

pub struct OperationsData {
    pub module_data: Arc<Mutex<CString>>
}

impl Clone for OperationsData {
    fn clone(&self) -> Self {
        let module_data = Arc::clone(&self.module_data);
        Self {
            module_data
        }
    }
}

impl OperationsData {
    pub fn new() -> Self {
        let module_data = CString::new(include_str!("../../cuda/operations.ptx")).expect("failed to open .ptx file");
        let module_data = Arc::new(Mutex::new(module_data));
        Self {
            module_data
        }
    }
}

pub struct Operations {
    pub module: Module,
    pub context: Context,
}

impl Operations {
    pub fn new(data: &OperationsData) -> Self {
        rustacuda::init(CudaFlags::empty()).expect("error initializing rustacuda");
    
        let device = Device::get_device(0).expect("error getting device");
        let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device).expect("error creating context");

        let module_data = data.module_data.lock().unwrap();
        let module = Module::load_from_string(&module_data).expect("error creating module");
        Self {
            module,
            context
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
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.add<<<grid_size, block_size, 0, stream>>>(
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
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.sub<<<grid_size, block_size, 0, stream>>>(
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
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.mul<<<grid_size, block_size, 0, stream>>>(
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
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.divide<<<grid_size, block_size, 0, stream>>>(
                x.as_device_ptr(),
                y.as_device_ptr(),
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
        grid_size: u32,   // num blocks
        block_size: u32,  // num threads
    ) -> CudaResult<DeviceBuffer<Vec3>> {
        unsafe {
            let out = vec![Vec3::zero(); count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_add<<<grid_size, block_size, 0, stream>>>(
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
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<DeviceBuffer<Vec3>> {
        unsafe {
            let out = vec![Vec3::zero(); count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_sub<<<grid_size, block_size, 0, stream>>>(
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
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<DeviceBuffer<Vec3>> {
        unsafe {
            let out = vec![Vec3::zero(); count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_mul_scalar<<<grid_size, block_size, 0, stream>>>(
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
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<DeviceBuffer<Vec3>> {
        unsafe {
            let out = vec![Vec3::zero(); count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_div_scalar<<<grid_size, block_size, 0, stream>>>(
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
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.len<<<grid_size, block_size, 0, stream>>>(
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
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<DeviceBuffer<Vec3>> {
        unsafe {
            let out = vec![Vec3::zero(); count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_normalize<<<grid_size, block_size, 0, stream>>>(
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
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_get_x<<<grid_size, block_size, 0, stream>>>(
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
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_get_y<<<grid_size, block_size, 0, stream>>>(
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
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<DeviceBuffer<f32>> {
        unsafe {
            let out = vec![0.0_f32; count];
            let mut out = DeviceBuffer::from_slice_async(&out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_get_z<<<grid_size, block_size, 0, stream>>>(
                v.as_device_ptr(),
                out.as_device_ptr(),
                out.len()
            ))?;
            Ok(out)
        }
    }
}