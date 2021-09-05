use std::error::Error;
use std::path::Path;
use std::fs::File;
use std::io::prelude::*;
use std::time::{Instant};

#[macro_use]
extern crate rustacuda;
use rustacuda::prelude::*;
use rustacuda::error::CudaResult;

use std::rc::Rc;

use rand::prelude::*;

mod lib;
use lib::{Vec3, Operations, Camera};
use lib::cuda_types::{CudaVec3, CudaFloat, CudaBool};

fn idx_to_uv(idx: u32, width: u32, height: u32)   -> (f32, f32) {
    let x = idx % width;
    let y = height - 1 - idx / width;

    let mut rng = rand::thread_rng();
    let x = x as f32 + rng.gen::<f32>();
    let y = y as f32 + rng.gen::<f32>();

    (x as f32 / (width as f32 - 1.0), y as f32 / (height as f32 - 1.0))
}

fn draw_image(filename: &str) -> Result<(), Box<dyn Error>> {
    let path = Path::new(filename);

    let mut output = String::new();

    // image setting
    const ASPECT_RATIO: f32 = 16.0 / 9.0;
    const WIDTH: u32 = 400;
    const HEIGHT: u32 = (WIDTH as f32 / ASPECT_RATIO) as u32;
    const RAY_PER_PIXEL: usize = 3;
    let cam = Camera::new(Vec3::zero(), ASPECT_RATIO, 2.0, 1.0);

    // ready cuda operations
    const GRID_SIZE: u32 = 128;
    const BLOCK_SIZE: u32 = 512;
    let operations = Rc::new(Operations::new(GRID_SIZE, BLOCK_SIZE));

    // ready streams
    let stream_cnt = 16_usize;   // (WIDTH * HEIGHT) % stream_cnt shoud be 0
    let chunck_size = ((WIDTH*HEIGHT) as usize / stream_cnt) * RAY_PER_PIXEL;
    let mut streams = Vec::new();
    for _ in 0..stream_cnt {
        let stream = operations.create_stream().unwrap();
        streams.push(Rc::new(stream));
    }

    // render
    println!("start rendering!");
    let timer = Instant::now();

    // ready dirs, origins
    let mut dirs = Vec::new();
    let mut origs = Vec::new();
    for i in 0..WIDTH*HEIGHT {
        for _ in 0..RAY_PER_PIXEL {
            let uv = idx_to_uv(i, WIDTH, HEIGHT);
            let ray = cam.get_ray(uv);
            dirs.push(ray.dir.clone());
            origs.push(ray.orig.clone());
        }
    }

    // push operations
    let mut cols_device = Vec::new();
    
    for i in 0..stream_cnt {
        let rng = (chunck_size*i, chunck_size*(i + 1));
        let stream = &streams[i as usize];
        let cols = push_operations(&operations, stream, &dirs, &origs, rng)?;
        cols_device.push(cols);
    }
    for i in 0..stream_cnt {
        let stream = &streams[i];
        stream.synchronize()?;
    }
    let duration = timer.elapsed();
    println!("calculation complete!, {:?}", duration);

    // write to file
    output += &format!("P3\n{} {}\n255\n", WIDTH as u32, HEIGHT as u32)[..];
    let pixel_per_chunk = chunck_size / RAY_PER_PIXEL;
    for cols in cols_device {
        let cols_host = cols.to_vec()?;
        for i in 0..pixel_per_chunk {
            let mut col = Vec3::zero();
            for partial_col in &cols_host[i*RAY_PER_PIXEL..(i+1)*RAY_PER_PIXEL] {
                col = partial_col.add(col);
            }
            col = col.mul(1.0 / RAY_PER_PIXEL as f32);
            let r = (255.999 * col.x) as u32;
            let g = (255.999 * col.y) as u32;
            let b = (255.999 * col.z) as u32;
            output += &format!("{} {} {}\n", r, g, b);
        }
    }
    let mut file = File::create(&path)?;
    file.write_all(output.as_bytes())?;
    println!("write complete!");
    Ok(())
}

fn push_operations(
    operations: &Rc<Operations>, 
    stream: &Rc<Stream>, 
    dirs: &Vec<Vec3>, 
    origs: &Vec<Vec3>, 
    range: (usize, usize),
) -> CudaResult<CudaVec3> {
    let chunk_size = range.1 - range.0;

    // constants
    let mut point_fives = CudaFloat::from_vec(vec![0.5_f32; chunk_size], stream, operations)?;
    let mut one_vecs = CudaVec3::from_vec(vec![Vec3::new(1.0, 1.0, 1.0); chunk_size], stream, operations)?;

    let mut dirs = CudaVec3::from_slice(&dirs[range.0..range.1], stream, operations)?;
    let mut origs = CudaVec3::from_slice(&origs[range.0..range.1], stream, operations)?;

    let center_vec = [Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, -100.5, -1.0)];
    let radius_vec = [0.5_f32, 100.0_f32];
    let mut centers_vec = Vec::new();
    let mut radiuses_vec = Vec::new();
    for i in 0..center_vec.len() {
        centers_vec.push(CudaVec3::from_vec(vec![center_vec[i].clone(); chunk_size], stream, operations)?);
        radiuses_vec.push(CudaFloat::from_vec(vec![radius_vec[i]; chunk_size], stream, operations)?);
    }

    // background_cols
    let mut cols = operations.background_color(&mut dirs, stream, operations)?;

    // sphere collision
    let mut t_min = CudaFloat::from_vec(vec![0.0_f32; chunk_size], stream, operations)?;
    let mut t_max = CudaFloat::from_vec(vec![1e10_f32; chunk_size], stream, operations)?;
    let mut hit_anything = CudaBool::new_false(chunk_size, stream, operations)?;
    let mut normal_col = CudaVec3::from_vec(vec![Vec3::zero(); chunk_size], stream, operations)?;
    for i in 0..centers_vec.len() {
        let centers = &mut centers_vec[i];
        let radiuses = &mut radiuses_vec[i];
        let mut collision_info = operations.get_sphere_collision_info(&mut origs, &mut dirs, centers, radiuses, &mut t_min, &mut t_max, chunk_size, stream, operations)?;

        hit_anything = CudaBool::or(&mut hit_anything, &mut collision_info.0)?;
        t_max = CudaFloat::select(&mut collision_info.0, &mut collision_info.1, &mut t_max)?;

        let mut collision_point = operations.ray_at(&mut origs, &mut dirs, &mut collision_info.1)?;
        let mut new_normal = operations.get_outward_normal(&mut collision_point.sub(centers)?, &mut dirs)?;

        let mut new_normal_col = new_normal.add(&mut one_vecs)?.mul_scalar(&mut point_fives)?;
        normal_col = CudaVec3::select(&mut collision_info.0, &mut new_normal_col, &mut normal_col)?;
    }

    // println!("{:?}", hit_anything.to_vec());

    cols = CudaVec3::select(&mut hit_anything, &mut normal_col, &mut cols)?;
    Ok(cols)
}

fn main() -> Result<(), Box<dyn Error>> {
    draw_image("output.ppm")?;
    Ok(())
}
