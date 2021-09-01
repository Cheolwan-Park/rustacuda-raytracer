#[derive(Clone, DeviceCopy, Debug)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            x, y, z
        }
    }

    pub fn zero() -> Self {
        Self {
            x: 0_f32,
            y: 0_f32,
            z: 0_f32,
        }
    }

    pub fn len_squared(&self) -> f32 {
        Vec3::dot(self, self)
    }

    pub fn len(&self) -> f32 {
        self.len_squared().sqrt()
    }

    pub fn inv(&self) -> Vec3 {
        Self::new(-self.x, -self.y, -self.z)
    }

    pub fn unit(&self) -> Vec3 {
        self.mul(1.0 / self.len())
    }

    pub fn add(&self, other: Vec3) -> Vec3 {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    pub fn mul(&self, scalar: f32) -> Vec3 {
        Self::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }

    pub fn dot(a: &Self, b: &Self) -> f32 {
        a.x * b.x + a.y * b.y + a.z * b.z
    }
}
