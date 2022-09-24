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

impl TraceResult {
    pub fn is_farther_than(&self, other_t: f32) -> bool {
        match self {
            TraceResult::Miss => true,
            TraceResult::Hit { t, .. } => *t > other_t,
        }
    }

    pub fn is_closer_than(&self, other: &Self) -> bool {
        match self {
            TraceResult::Miss => false,
            TraceResult::Hit { t, .. } => match other {
                TraceResult::Miss => true,
                TraceResult::Hit { t: other_t, .. } => t < other_t,
            },
        }
    }
}

pub trait AccelerationStructure {
    fn intersect(&self, ray: &Ray, verts: &[Vertex], triangles: &[Triangle]) -> TraceResult;

    fn get_name(&self) -> &str;

    fn get_statistics(&self) -> StatisticsStore;
}
