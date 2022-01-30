use crate::mesh::Vertex;
use crate::raytracer::triangle::Triangle;
use crate::raytracer::Ray;
use super::statistics::StatisticsStore;

pub enum TraceResult {
    Miss,
    Hit {
        triangle_index: i32,
        t: f32,
        u: f32,
        v: f32,
    },
}

pub trait AccelerationStructure {
    fn intersect(&self, ray: &Ray, verts: &[Vertex], triangles: &[Triangle]) -> TraceResult;

    fn get_name(&self) -> &str;

    fn get_statistics(&self) -> StatisticsStore;
}