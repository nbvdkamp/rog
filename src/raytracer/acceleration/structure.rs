use crate::mesh::Vertex;
use crate::raytracer::triangle::Triangle;
use crate::raytracer::Ray;

pub trait AccelerationStructure {
    fn new(verts: &[Vertex], triangles: &[Triangle]) -> Self;
    fn intersect(&self, ray: &Ray) -> Vec<usize>;
}