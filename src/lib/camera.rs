use super::Vec3;
use super::Ray;

pub struct Camera {
    origin: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    lower_left_corner: Vec3,
}

impl Camera {
    pub fn new(origin: Vec3, aspect_ratio: f32, viewport_height: f32, focal_len: f32) -> Self {
        let viewport_width = viewport_height * aspect_ratio;
        let horizontal = Vec3::new(viewport_width, 0.0, 0.0);
        let vertical = Vec3::new(0.0, viewport_height, 0.0);
        let lower_left_corner = origin.add(horizontal.mul(0.5).inv()).add(vertical.mul(0.5).inv()).add(Vec3::new(0.0, 0.0, focal_len).inv());

        Self {
            origin,
            horizontal,
            vertical,
            lower_left_corner
        }
    }

    pub fn get_ray(&self, uv: (f32, f32)) -> Ray {
        let dir = self.lower_left_corner.add(self.horizontal.mul(uv.0)).add(self.vertical.mul(uv.1)).add(self.origin.inv());
        Ray::new(self.origin.clone(), dir)
    }
}