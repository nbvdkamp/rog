use cgmath::Point3;

use super::statistics::StatisticsStore;
use crate::{
    barycentric::Barycentric,
    mesh::Instance,
    raytracer::{triangle::Triangle, Ray},
};

pub enum TraceResult {
    Miss,
    Hit {
        triangle_index: u32,
        t: f32,
        barycentric: Barycentric,
    },
}

pub enum TraceResultMesh<'a> {
    Miss,
    Hit {
        instance: &'a Instance,
        triangle_index: u32,
        t: f32,
        barycentric: Barycentric,
    },
}

impl TraceResult {
    pub fn is_farther_than(&self, other_t: f32) -> bool {
        match self {
            Self::Miss => true,
            Self::Hit { t, .. } => *t > other_t,
        }
    }

    pub fn is_closer_than(&self, other: &Self) -> bool {
        match self {
            Self::Miss => false,
            Self::Hit { t, .. } => match other {
                Self::Miss => true,
                Self::Hit { t: other_t, .. } => t < other_t,
            },
        }
    }

    pub fn with_instance(self, instance: &Instance) -> TraceResultMesh {
        match self {
            Self::Miss => TraceResultMesh::Miss,
            Self::Hit {
                triangle_index,
                t,
                barycentric,
            } => TraceResultMesh::Hit {
                instance,
                triangle_index,
                t,
                barycentric,
            },
        }
    }
}

impl<'a> TraceResultMesh<'a> {
    pub fn is_closer_than(&self, other: &Self) -> bool {
        match self {
            Self::Miss => false,
            Self::Hit { t, .. } => match other {
                Self::Miss => true,
                Self::Hit { t: other_t, .. } => t < other_t,
            },
        }
    }
}

pub trait AccelerationStructure {
    fn intersect(&self, ray: &Ray, verts: &[Point3<f32>], triangles: &[Triangle]) -> TraceResult;

    fn get_statistics(&self) -> StatisticsStore;
}
