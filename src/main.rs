#[macro_use]
extern crate rustacuda;
use rustacuda::prelude::*;
use rustacuda::error::CudaResult;

use std::error::Error;
use std::path::Path;
use std::fs::File;
use std::io::prelude::*;
use std::time::{Instant};

mod lib;
use lib::{Vec3, Operations, OperationsData, Camera};

fn idx_to_uv(idx: u32, width: u32, height: u32) -> (f32, f32) {
    let x = idx % width;
    let y = height - 1 - idx / width;

    (x as f32 / (width as f32 - 1.0), y as f32 / (height as f32 - 1.0))
}

fn draw_image(filename: &str) -> Result<(), Box<dyn Error>> {
    let path = Path::new(filename);

    let mut output = String::new();

    // image setting
    const ASPECT_RATIO: f32 = 16.0 / 9.0;
    const WIDTH: u32 = 1920;
    const HEIGHT: u32 = (WIDTH as f32 / ASPECT_RATIO) as u32;
    let cam = Camera::new(Vec3::zero(), ASPECT_RATIO, 2.0, 1.0);

    // ready cuda operations
    let operations = Operations::new(&OperationsData::new());

    // ready streams
    let stream_cnt = 8_usize;   // (WIDTH * HEIGHT) % stream_cnt shoud be 0
    let chunck_size = (WIDTH*HEIGHT) as usize / stream_cnt;
    let mut streams = Vec::new();
    for _ in 0..stream_cnt {
        let stream = operations.create_stream().unwrap();
        streams.push(stream);
    }

    // render
    println!("start rendering!");
    let timer = Instant::now();

    // ready dirs, origins
    let mut dirs = Vec::new();
    let mut origs = Vec::new();
    for i in 0..WIDTH*HEIGHT {
        let uv = idx_to_uv(i, WIDTH, HEIGHT);
        let ray = cam.get_ray(uv);
        dirs.push(ray.dir.clone());
        origs.push(ray.orig.clone());
    }

    // push operations
    const GRID_SIZE: u32 = 16;
    const BLOCK_SIZE: u32 = 256;
    let mut cols_device = Vec::new();
    
    for i in 0..stream_cnt {
        let rng = (chunck_size*i, chunck_size*(i + 1));
        let stream = &streams[i as usize];
        let cols = push_operations(&operations, stream, &dirs, &origs, rng, GRID_SIZE, BLOCK_SIZE)?;
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
    for cols in cols_device {
        let mut cols_host = vec![Vec3::zero(); chunck_size];
        cols.copy_to(&mut cols_host[..])?;
        for col in cols_host {
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
    operations: &Operations, 
    stream: &Stream, 
    dirs: &Vec<Vec3>, 
    _origs: &Vec<Vec3>, 
    range: (usize, usize), 
    grid_size: u32, 
    block_size: u32
) -> CudaResult<DeviceBuffer<Vec3>> {
    let chunck_size = range.1 - range.0;

    // constants
    let ones = vec![1.0_f32; chunck_size];
    let point_fives = vec![0.5_f32; chunck_size];

    let colors1 = vec![Vec3::new(0.5, 0.7, 1.0); chunck_size];
    let colors2 = vec![Vec3::new(1.0, 1.0, 1.0); chunck_size];

    // move to device
    let mut vec = Operations::slice_to_device(&dirs[range.0..range.1], stream)?;

    // push operations
    let mut vec = operations.vec3_normalize(&mut vec, chunck_size, stream, grid_size, block_size)?;
    let mut t = operations.vec3_get_y(&mut vec, chunck_size, &stream, grid_size, block_size)?;

    let mut val = Operations::slice_to_device(&ones[..], &stream)?;
    let mut t = operations.add(&mut t, &mut val, chunck_size, &stream, grid_size, block_size)?;


    let mut val = Operations::slice_to_device(&point_fives[..], &stream)?;
    let mut t = operations.mul(&mut t, &mut val, chunck_size, &stream, grid_size, block_size)?;

    let mut vec = Operations::slice_to_device(&colors1[..], &stream)?;
    let mut cols = operations.vec3_mul_scalar(&mut vec, &mut t, chunck_size, &stream, grid_size, block_size)?;

    let mut val = Operations::slice_to_device(&ones[..], &stream)?;
    let mut t = operations.sub(&mut val, &mut t, chunck_size, &stream, grid_size, block_size)?;

    let mut vec = Operations::slice_to_device(&colors2[..], &stream)?;
    let mut cols2 = operations.vec3_mul_scalar(&mut vec, &mut t, chunck_size, &stream, grid_size, block_size)?;
    let cols = operations.vec3_add(&mut cols, &mut cols2, chunck_size, &stream, grid_size, block_size)?;

    Ok(cols)
}

fn main() -> Result<(), Box<dyn Error>> {
    draw_image("output.ppm")?;
    Ok(())
}
