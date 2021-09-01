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
        x: &[f32],
        y: &[f32],
        stream: &Stream,
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<Vec<f32>> {
        let mut out = vec!(0.0_f32; x.len());
        unsafe {
            let mut x_device = DeviceBuffer::from_slice_async(x, stream)?;
            let mut y_device = DeviceBuffer::from_slice_async(y, stream)?;
            let mut out_device = DeviceBuffer::from_slice_async(&mut out[..], stream)?;
            
            let module = &self.module;
            launch!(module.add<<<grid_size, block_size, 0, stream>>>(
                x_device.as_device_ptr(),
                y_device.as_device_ptr(),
                out_device.as_device_ptr(),
                out.len()
            ))?;
            stream.synchronize()?;

            out_device.copy_to(&mut out[..])?;
        }
        Ok(out)
    }

    pub fn sub(
        &self,
        x: &[f32],
        y: &[f32],
        stream: &Stream,
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<Vec<f32>> {
        let mut out = vec!(0.0_f32; x.len());
        unsafe {
            let mut x_device = DeviceBuffer::from_slice_async(x, stream)?;
            let mut y_device = DeviceBuffer::from_slice_async(y, stream)?;
            let mut out_device = DeviceBuffer::from_slice_async(&mut out[..], stream)?;
            
            let module = &self.module;
            launch!(module.sub<<<grid_size, block_size, 0, stream>>>(
                x_device.as_device_ptr(),
                y_device.as_device_ptr(),
                out_device.as_device_ptr(),
                out.len()
            ))?;
            stream.synchronize()?;

            out_device.copy_to(&mut out[..])?;
        }
        Ok(out)
    }

    pub fn mul(
        &self,
        x: &[f32],
        y: &[f32],
        stream: &Stream,
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<Vec<f32>> {
        let mut out = vec!(0.0_f32; x.len());
        unsafe {
            let mut x_device = DeviceBuffer::from_slice_async(x, stream)?;
            let mut y_device = DeviceBuffer::from_slice_async(y, stream)?;
            let mut out_device = DeviceBuffer::from_slice_async(&mut out[..], stream)?;
            
            let module = &self.module;
            launch!(module.mul<<<grid_size, block_size, 0, stream>>>(
                x_device.as_device_ptr(),
                y_device.as_device_ptr(),
                out_device.as_device_ptr(),
                out.len()
            ))?;
            stream.synchronize()?;

            out_device.copy_to(&mut out[..])?;
        }
        Ok(out)
    }

    pub fn div(
        &self,
        x: &[f32],
        y: &[f32],
        stream: &Stream,
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<Vec<f32>> {
        let mut out = vec!(0.0_f32; x.len());
        unsafe {
            let mut x_device = DeviceBuffer::from_slice_async(x, stream)?;
            let mut y_device = DeviceBuffer::from_slice_async(y, stream)?;
            let mut out_device = DeviceBuffer::from_slice_async(&mut out[..], stream)?;
            
            let module = &self.module;
            launch!(module.divide<<<grid_size, block_size, 0, stream>>>(
                x_device.as_device_ptr(),
                y_device.as_device_ptr(),
                out_device.as_device_ptr(),
                out.len()
            ))?;
            stream.synchronize()?;

            out_device.copy_to(&mut out[..])?;
        }
        Ok(out)
    }

    pub fn vec3_add(
        &self,
        x: &[Vec3], 
        y: &[Vec3],
        stream: &Stream, 
        grid_size: u32,   // num blocks
        block_size: u32,  // num threads
    ) -> CudaResult<Vec<Vec3>> {
        let mut out = vec!(Vec3::zero(); x.len());
        unsafe {
            let mut x_device = DeviceBuffer::from_slice_async(x, stream)?;
            let mut y_device = DeviceBuffer::from_slice_async(y, stream)?;
            let mut out_device = DeviceBuffer::from_slice_async(&mut out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_add<<<grid_size, block_size, 0, stream>>>(
                x_device.as_device_ptr(),
                y_device.as_device_ptr(),
                out_device.as_device_ptr(),
                out.len()
            ))?;
            stream.synchronize()?;

            out_device.copy_to(&mut out[..])?;
        }
        Ok(out)
    }

    pub fn vec3_sub(
        &self,
        x: &[Vec3], 
        y: &[Vec3],
        stream: &Stream, 
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<Vec<Vec3>> {
        let mut out = vec!(Vec3::zero(); x.len());
        unsafe {
            let mut x_device = DeviceBuffer::from_slice_async(x, stream)?;
            let mut y_device = DeviceBuffer::from_slice_async(y, stream)?;
            let mut out_device = DeviceBuffer::from_slice_async(&mut out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_sub<<<grid_size, block_size, 0, stream>>>(
                x_device.as_device_ptr(),
                y_device.as_device_ptr(),
                out_device.as_device_ptr(),
                out.len()
            ))?;
            stream.synchronize()?;

            out_device.copy_to(&mut out[..])?;
        }
        Ok(out)
    }

    pub fn vec3_mul_scalar(
        &self,
        v: &[Vec3],
        scalar: &[f32],
        stream: &Stream,
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<Vec<Vec3>> {
        let mut out = vec!(Vec3::zero(); v.len());
        unsafe {
            let mut v_device = DeviceBuffer::from_slice_async(v, stream)?;
            let mut scalar_device = DeviceBuffer::from_slice_async(scalar, stream)?;
            let mut out_device = DeviceBuffer::from_slice_async(&mut out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_mul_scalar<<<grid_size, block_size, 0, stream>>>(
                v_device.as_device_ptr(),
                scalar_device.as_device_ptr(),
                out_device.as_device_ptr(),
                out.len()
            ))?;
            stream.synchronize()?;

            out_device.copy_to(&mut out[..])?;
        }
        Ok(out)
    }

    pub fn vec3_div_scalar(
        &self,
        v: &[Vec3],
        scalar: &[f32],
        stream: &Stream,
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<Vec<Vec3>> {
        let mut out = vec!(Vec3::zero(); v.len());
        unsafe {
            let mut v_device = DeviceBuffer::from_slice_async(v, stream)?;
            let mut scalar_device = DeviceBuffer::from_slice_async(scalar, stream)?;
            let mut out_device = DeviceBuffer::from_slice_async(&mut out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_div_scalar<<<grid_size, block_size, 0, stream>>>(
                v_device.as_device_ptr(),
                scalar_device.as_device_ptr(),
                out_device.as_device_ptr(),
                out.len()
            ))?;
            stream.synchronize()?;

            out_device.copy_to(&mut out[..])?;
        }
        Ok(out)
    }

    pub fn vec3_len(
        &self,
        v: &[Vec3],
        stream: &Stream,
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<()> {
        let mut out = vec!(0.0_f32; v.len());
        unsafe {
            let mut v_device = DeviceBuffer::from_slice_async(v, stream)?;
            let mut out_device = DeviceBuffer::from_slice_async(& mut out[..], stream)?;
            
            let module = &self.module;
            launch!(module.len<<<grid_size, block_size, 0, stream>>>(
                v_device.as_device_ptr(),
                out_device.as_device_ptr(),
                out.len()
            ))?;
            stream.synchronize()?;

            out_device.copy_to(&mut out[..])?;
        }
        Ok(())
    }

    pub fn vec3_normalize(
        &self,
        v: &[Vec3],
        stream: &Stream,
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<Vec<Vec3>> {
        let mut out = vec!(Vec3::zero(); v.len());
        unsafe {
            let mut v_device = DeviceBuffer::from_slice_async(v, stream)?;
            let mut out_device = DeviceBuffer::from_slice_async(&mut out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_normalize<<<grid_size, block_size, 0, stream>>>(
                v_device.as_device_ptr(),
                out_device.as_device_ptr(),
                out.len()
            ))?;
            stream.synchronize()?;

            out_device.copy_to(&mut out[..])?;
        }
        Ok(out)
    }

    pub fn vec3_get_x(
        &self,
        v: &[Vec3],
        stream: &Stream,
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<()> {
        let mut out = vec!(0.0_f32; v.len());
        unsafe {
            let mut v_device = DeviceBuffer::from_slice_async(v, stream)?;
            let mut out_device = DeviceBuffer::from_slice_async(&mut out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_get_x<<<grid_size, block_size, 0, stream>>>(
                v_device.as_device_ptr(),
                out_device.as_device_ptr(),
                out.len()
            ))?;
            stream.synchronize()?;

            out_device.copy_to(&mut out[..])?;
        }
        Ok(())
    }

    pub fn vec3_get_y(
        &self,
        v: &[Vec3],
        stream: &Stream,
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<Vec<f32>> {
        let mut out = vec!(0.0_f32; v.len());
        unsafe {
            let mut v_device = DeviceBuffer::from_slice_async(v, stream)?;
            let mut out_device = DeviceBuffer::from_slice_async(&mut out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_get_y<<<grid_size, block_size, 0, stream>>>(
                v_device.as_device_ptr(),
                out_device.as_device_ptr(),
                out.len()
            ))?;
            stream.synchronize()?;

            out_device.copy_to(&mut out[..])?;
        }
        Ok(out)
    }

    pub fn vec3_get_z(
        &self,
        v: &[Vec3],
        stream: &Stream,
        grid_size: u32,
        block_size: u32,
    ) -> CudaResult<()> {
        let mut out = vec!(0.0_f32; v.len());
        unsafe {
            let mut v_device = DeviceBuffer::from_slice_async(v, stream)?;
            let mut out_device = DeviceBuffer::from_slice_async(&mut out[..], stream)?;
            
            let module = &self.module;
            launch!(module.vec3_get_z<<<grid_size, block_size, 0, stream>>>(
                v_device.as_device_ptr(),
                out_device.as_device_ptr(),
                out.len()
            ))?;
            stream.synchronize()?;

            out_device.copy_to(&mut out[..])?;
        }
        Ok(())
    }
}