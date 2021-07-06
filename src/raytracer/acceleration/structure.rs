use crate::mesh::Vertex;
use crate::raytracer::triangle::Triangle;
use crate::raytracer::Ray;

pub trait AccelerationStructure {
    fn new(verts: &Vec<Vertex>, triangles: &Vec<Triangle>) -> Self;
    fn intersect(&self, ray: Ray) -> Vec<usize>;
}