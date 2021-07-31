use crate::mesh::Vertex;
use crate::raytracer::triangle::Triangle;
use crate::raytracer::Ray;

use cgmath::Point3;

pub enum TraceResult {
    Miss,
    // Represents the triangle index and hit position
    Hit(i32, Point3<f32>),
}

pub trait AccelerationStructure {
    fn intersect(&self, ray: &Ray, verts: &[Vertex], triangles: &[Triangle]) -> TraceResult;
}