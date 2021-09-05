use rustacuda::prelude::*;
use rustacuda::error::CudaResult;

use super::{Vec3, Operations};

impl Operations {
    pub fn background_color(
        &self,
        dirs: &mut DeviceBuffer<Vec3>,
        chunk_size: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<Vec3>> {
        // constants
        let ones = vec![1.0_f32; chunk_size];
        let point_fives = vec![0.5_f32; chunk_size];
        let colors1 = vec![Vec3::new(0.5, 0.7, 1.0); chunk_size];
        let colors2 = vec![Vec3::new(1.0, 1.0, 1.0); chunk_size];
        
        // move to device
        let mut ones = Operations::slice_to_device(&ones[..], stream)?;
        let mut point_fives = Operations::slice_to_device(&point_fives[..], stream)?;
        let mut colors1 = Operations::slice_to_device(&colors1[..], stream)?;
        let mut colors2 = Operations::slice_to_device(&colors2[..], stream)?;
    
        // push operations
        let mut vec = self.vec3_normalize(dirs, chunk_size, stream)?;
        let mut t = self.vec3_get_y(&mut vec, chunk_size, stream)?;
        t = self.add(&mut t, &mut ones, chunk_size, stream)?;
        t = self.mul(&mut t, &mut point_fives, chunk_size, stream)?;
    
        let mut cols = self.vec3_mul_scalar(&mut colors1, &mut t, chunk_size, stream)?;
        t = self.sub(&mut ones, &mut t, chunk_size, stream)?;
        let mut cols2 = self.vec3_mul_scalar(&mut colors2, &mut t, chunk_size, stream)?;
        
        self.vec3_add(&mut cols, &mut cols2, chunk_size, stream)
    }

    
    pub fn get_sphere_collision_info( // returns (is_collide, t)
        &self,
        origs: &mut DeviceBuffer<Vec3>,
        dirs: &mut DeviceBuffer<Vec3>,
        centers: &mut DeviceBuffer<Vec3>,
        radiuses: &mut DeviceBuffer<f32>,
        t_min: &mut DeviceBuffer<f32>,
        t_max: &mut DeviceBuffer<f32>,
        chunk_size: usize,
        stream: &Stream,
    ) -> CudaResult<(DeviceBuffer<f32>, DeviceBuffer<f32>)> {
        // constants
        let twos = vec![2.0_f32; chunk_size];
    
        // move to device
        let mut twos = Operations::slice_to_device(&twos[..], stream)?;
    
        // push operations
        let mut oc = self.vec3_sub(origs, centers, chunk_size, stream)?;
            
        let mut a = self.vec3_len_squared(dirs, chunk_size, stream)?;
    
        let mut half_b = self.vec3_dot(&mut oc, dirs, chunk_size, stream)?;
        
        let mut c = self.vec3_len_squared(&mut oc, chunk_size, stream)?;
        let mut c2 = self.pow(radiuses, &mut twos, chunk_size, stream)?;
        c = self.sub(&mut c, &mut c2, chunk_size, stream)?;
    
        let mut d = self.pow(&mut half_b, &mut twos, chunk_size, stream)?;
        let mut ac = self.mul(&mut a, &mut c, chunk_size, stream)?;
        d = self.sub(&mut d, &mut ac, chunk_size, stream)?;
    
        let mut is_pos = self.is_positive(&mut d, chunk_size, stream)?;
        
        half_b = self.inv(&mut half_b, chunk_size, stream)?;
        d = self.mul(&mut d, &mut is_pos, chunk_size, stream)?; // make negative values 0
        d = self.sqrt(&mut d, chunk_size, stream)?;
        
        let mut t = self.sub(&mut half_b, &mut d, chunk_size, stream)?;
        t = self.div(&mut t, &mut a, chunk_size, stream)?;
        
        let mut t2 = self.add(&mut half_b, &mut d, chunk_size, stream)?;
        t2 = self.div(&mut t2, &mut a, chunk_size, stream)?;

        let mut in_range = self.compare(&mut t, t_min, chunk_size, stream)?; 
        let mut in_range_ = self.compare(t_max, &mut t, chunk_size, stream)?;
        in_range = self.and(&mut in_range, &mut in_range_, chunk_size, stream)?;
        
        let mut in_range2 = self.compare(&mut t2, t_min, chunk_size, stream)?; 
        let mut in_range2_ = self.compare(t_max, &mut t2, chunk_size, stream)?;
        in_range2 = self.and(&mut in_range2, &mut in_range2_, chunk_size, stream)?;

        t = self.select(&mut in_range, &mut t, &mut t2, chunk_size, stream)?;
        in_range = self.or(&mut in_range, &mut in_range2, chunk_size, stream)?;
        is_pos = self.and(&mut is_pos, &mut in_range, chunk_size, stream)?;
    
        Ok((is_pos, t))
    }

    pub fn ray_at(
        &self,
        origs: &mut DeviceBuffer<Vec3>,
        dirs: &mut DeviceBuffer<Vec3>,
        t: &mut DeviceBuffer<f32>,
        chunk_size: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<Vec3>> {
        let mut delta = self.vec3_mul_scalar(dirs, t, chunk_size, stream)?;
        self.vec3_add(&mut delta, origs, chunk_size, stream)
    }

    pub fn get_outward_normal(      // returns (is_front_face, outward_normal)
        &self,
        normals: &mut DeviceBuffer<Vec3>,
        ray_dirs: &mut DeviceBuffer<Vec3>,
        chunk_size: usize,
        stream: &Stream,
    ) -> CudaResult<DeviceBuffer<Vec3>> {
        let mut front_face = self.vec3_dot(normals, ray_dirs, chunk_size, stream)?;
        front_face = self.is_negative(&mut front_face, chunk_size, stream)?;

        let mut inv_normals = self.vec3_inv(normals, chunk_size, stream)?;
        let normals = self.vec3_select(&mut front_face, normals, &mut inv_normals, chunk_size, stream)?;
        
        Ok(normals)
    }

    pub fn println_floats(
        &self,
        val: &DeviceBuffer<f32>,
        count: usize,
        stream: &Stream
    ) -> CudaResult<()> {
        let mut host = vec![0.0_f32; count];
        stream.synchronize()?;
        val.copy_to(&mut host)?;
        println!("{:?}", host);
        Ok(())
    }

    pub fn println_vec3s(
        &self,
        val: &DeviceBuffer<Vec3>,
        count: usize,
        stream: &Stream
    ) -> CudaResult<()> {
        let mut host = vec![Vec3::zero(); count];
        stream.synchronize()?;
        val.copy_to(&mut host)?;
        println!("{:?}", host);
        Ok(())
    }
}