use rustacuda::prelude::*;
use rustacuda::error::CudaResult;

use std::rc::Rc;

use super::{Vec3, Operations};
use super::cuda_types::{CudaVec3, CudaFloat, CudaBool};

impl Operations {
    pub fn background_color(
        &self,
        dirs: &mut CudaVec3,
        stream: &Rc<Stream>,
        operations: &Rc<Operations>,
    ) -> CudaResult<CudaVec3> {
        let chunk_size = dirs.len();

        // constants
        let mut ones = CudaFloat::from_vec(vec![1.0_f32; chunk_size], stream, operations)?;
        let mut point_fives = CudaFloat::from_vec(vec![0.5_f32; chunk_size], stream, operations)?;
        let mut colors1 = CudaVec3::from_vec(vec![Vec3::new(0.5, 0.7, 1.0); chunk_size], stream, operations)?;
        let mut colors2 = CudaVec3::from_vec(vec![Vec3::new(1.0, 1.0, 1.0); chunk_size], stream, operations)?;
    
        // calculation
        let mut t = dirs.normalized()?.y()?.add(&mut ones)?.mul(&mut point_fives)?;

        let mut cols1 = colors1.mul_scalar(&mut t)?;
        let mut cols2 = colors2.mul_scalar(&mut ones.sub(&mut t)?)?;
        
        cols1.add(&mut cols2)
    }

    
    pub fn get_sphere_collision_info( // returns (is_collide, t)
        &self,
        orig: &mut CudaVec3,
        dir: &mut CudaVec3,
        center: &mut CudaVec3,
        radius: &mut CudaFloat,
        t_min: &mut CudaFloat,
        t_max: &mut CudaFloat,
        chunk_size: usize,
        stream: &Rc<Stream>,
        operations: &Rc<Operations>,
    ) -> CudaResult<(CudaBool, CudaFloat)> {
        // constants
        let mut two = CudaFloat::from_vec(vec![2.0_f32; chunk_size], stream, operations)?;
    
        // calculation
        let mut oc = orig.sub(center)?;                                                 // oc = orig - center
        
        let mut a = dir.vec_len_squared()?;                                             // a = dot(dir, dir)
        let mut half_b = CudaVec3::dot(&mut oc, dir)?;                                  // half_b = dot(oc, dir)
        let mut c = oc.vec_len_squared()?.sub(&mut CudaFloat::pow(radius, &mut two)?)?; // c = dot(oc, oc) - radius^2

        let mut d = CudaFloat::pow(&mut half_b, &mut two)?.sub(&mut a.mul(&mut c)?)?;   // half_b^2 - ac
        
        let is_pos = d.is_positive()?;

        let mut is_pos = is_pos.as_float();
        d = d.mul(&mut is_pos)?.sqrt()?;    // make negative values 0 (to prevent NaN)
        half_b = half_b.inv()?;

        let mut t = half_b.sub(&mut d)?.div(&mut a)?;

        let mut t2 = half_b.add(&mut d)?.div(&mut a)?;

        let mut in_range = CudaBool::and(&mut CudaFloat::compare(&mut t, t_min)?, &mut CudaFloat::compare(t_max, &mut t)?)?;
        let mut in_range2 = CudaBool::and(&mut CudaFloat::compare(&mut t2, t_min)?, &mut CudaFloat::compare(t_max, &mut t2)?)?;

        t = CudaFloat::select(&mut in_range, &mut t, &mut t2)?;
        
        let mut is_pos = is_pos.as_bool();
        is_pos = CudaBool::and(&mut is_pos, &mut CudaBool::or(&mut in_range, &mut in_range2)?)?;
    
        Ok((is_pos, t))
    }

    pub fn ray_at(
        &self,
        orig: &mut CudaVec3,
        dir: &mut CudaVec3,
        t: &mut CudaFloat,
    ) -> CudaResult<CudaVec3> {
        orig.add(&mut dir.mul_scalar(t)?)
    }

    pub fn get_outward_normal(      // returns (is_front_face, outward_normal)
        &self,
        normal: &mut CudaVec3,
        ray_dir: &mut CudaVec3,
    ) -> CudaResult<CudaVec3> {
        let mut front_face = CudaVec3::dot(normal, ray_dir)?.is_negative()?;
        let mut inv_normal = normal.inv()?;
        CudaVec3::select(&mut front_face, normal, &mut inv_normal)
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