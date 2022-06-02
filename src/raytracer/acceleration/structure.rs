use super::statistics::StatisticsStore;
use crate::{
    mesh::Vertex,
    raytracer::{triangle::Triangle, Ray},
};

pub enum TraceResult {
    Miss,
    Hit {
        triangle_index: u32,
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
