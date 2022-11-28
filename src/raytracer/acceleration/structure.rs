use cgmath::Point3;

use super::statistics::StatisticsStore;
use crate::raytracer::{aabb::BoundingBox, triangle::Triangle, Ray};

pub enum TraceResult {
    Miss,
    Hit {
        triangle_index: u32,
        t: f32,
        u: f32,
        v: f32,
    },
}

pub enum TraceResultMesh {
    Miss,
    Hit {
        mesh_index: u32,
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

    pub fn with_mesh_index(self, mesh_index: usize) -> TraceResultMesh {
        match self {
            TraceResult::Miss => TraceResultMesh::Miss,
            TraceResult::Hit {
                triangle_index,
                t,
                u,
                v,
            } => TraceResultMesh::Hit {
                mesh_index: mesh_index as u32,
                triangle_index,
                t,
                u,
                v,
            },
        }
    }
}

pub trait AccelerationStructure {
    fn intersect(&self, ray: &Ray, verts: &[Point3<f32>], triangles: &[Triangle]) -> TraceResult;

    fn bounds(&self) -> BoundingBox;

    fn get_statistics(&self) -> StatisticsStore;
}
