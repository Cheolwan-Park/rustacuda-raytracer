#[macro_use]
extern crate rustacuda;

use std::error::Error;
use std::path::Path;
use std::fs::File;
use std::io::prelude::*;
use std::thread;
use std::sync::{Mutex, Arc};
use std::time::{Instant};

use rustacuda::error::CudaResult;

mod lib;
use lib::{Vec3, Ray, Operations, OperationsData, Camera};

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
    const WIDTH: u32 = 400;
    const HEIGHT: u32 = (WIDTH as f32 / ASPECT_RATIO) as u32;
    let cam = Camera::new(Vec3::zero(), ASPECT_RATIO, 2.0, 1.0);

    // processing setting
    let thread_count = 1;       // width * height % thread_count shoud be 0

    // render
    println!("start rendering!");
    output += &format!("P3\n{} {}\n255\n", WIDTH as u32, HEIGHT as u32)[..];
    let chunck_size = WIDTH*HEIGHT / thread_count;

    let colors = Arc::new(Mutex::new(vec!((0_u32, 0_u32, 0_u32); (WIDTH*HEIGHT) as usize)));
    let cam = Arc::new(Mutex::new(cam));
    let operations_data = OperationsData::new();
    let mut handles = Vec::new();
    
    let timer = Instant::now();
    for i in 0..thread_count {
        let colors = Arc::clone(&colors);
        let cam = Arc::clone(&cam);
        let operations_data = operations_data.clone();
        let handle = thread::spawn(move || {
            let operations = Operations::new(&operations_data);
            match draw_partial(&operations, WIDTH, HEIGHT, &cam, (chunck_size*i, chunck_size*(i+1)), &colors) {
                Err(e) => {
                    println!("error occured while drawing: {:?}", e);
                },
                Ok(_) => {}
            };
        }); 
        handles.push(handle);
    }
    for handle in handles {
        handle.join().expect("some threads cannot be joined");
    }
    let duration = timer.elapsed();
    println!("calculation complete!, {:?}", duration);
    let colors = colors.lock().unwrap();
    for i in 0..WIDTH*HEIGHT {
        let idx = i as usize;
        output += &format!("{} {} {}\n", colors[idx].0, colors[idx].1, colors[idx].2);
    }
    println!("write complete!");

    let mut file = File::create(&path)?;
    file.write_all(output.as_bytes())?;
    Ok(())
}

fn draw_partial(operations: &Operations, width: u32, height: u32, camera: &Arc<Mutex<Camera>>, range: (u32, u32), colors: &Arc<Mutex<Vec<(u32, u32, u32)>>>) -> CudaResult<()> {
    let mut rays = Vec::new();
    let camera = camera.lock().unwrap();
    for i in range.0..range.1 {
        let uv = idx_to_uv(i, width, height);
        let ray = camera.get_ray(uv);
        rays.push(ray);
    }
    calculate_colors(operations, &rays, range, colors)?;
    Ok(())
}

fn calculate_colors(operations: &Operations, rays: &[Ray], range: (u32, u32), colors: &Arc<Mutex<Vec<(u32, u32, u32)>>>) -> CudaResult<()> {
    let stream = operations.create_stream().unwrap();

    let mut origins = Vec::new();
    let mut dirs = Vec::new();
    for i in 0..rays.len() {
        origins.push(rays[i].orig.clone());
        dirs.push(rays[i].dir.clone());
    }

    const GRID_SIZE: u32 = 32;
    const BLOCK_SIZE: u32 = 256;

    let units = operations.vec3_normalize(&dirs[..], &stream, GRID_SIZE, BLOCK_SIZE)?;
    let t = operations.vec3_get_y(&units[..], &stream, GRID_SIZE, BLOCK_SIZE)?;
    let t = operations.add(&t[..], &vec![1.0_f32; t.len()][..], &stream, GRID_SIZE, BLOCK_SIZE)?;
    let t = operations.mul(&t[..], &vec![0.5_f32; t.len()][..], &stream, GRID_SIZE, BLOCK_SIZE)?;
    let cols = operations.vec3_mul_scalar(&vec![Vec3::new(0.5, 0.7, 1.0); t.len()], &t[..], &stream, GRID_SIZE, BLOCK_SIZE)?;
    let t = operations.sub(&vec![1.0_f32; t.len()][..], &t[..], &stream, GRID_SIZE, BLOCK_SIZE)?;
    let cols2 = operations.vec3_mul_scalar(&vec![Vec3::new(1.0, 1.0, 1.0); t.len()][..], &t[..], &stream, GRID_SIZE, BLOCK_SIZE)?;
    let cols = operations.vec3_add(&cols[..], &cols2[..], &stream, GRID_SIZE, BLOCK_SIZE)?;

    let mut colors_arr = colors.lock().unwrap();
    for i in 0..origins.len() {
        let col = &cols[i];
        // let unit = dirs[i].unit();
        // let t = 0.5 * (unit.y + 1.0);
        // let col = Vec3::new(0.5, 0.7, 1.0).mul(t).add(Vec3::new(1.0, 1.0, 1.0).mul(1.0 - t));

        let r = (255.999 * col.x) as u32;
        let g = (255.999 * col.y) as u32;
        let b = (255.999 * col.z) as u32;
        colors_arr[i + range.0 as usize] = (r, g, b);
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    draw_image("output.ppm")?;
    Ok(())
}
