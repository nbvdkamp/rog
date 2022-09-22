pub mod statistics;
pub mod structure;
// pub mod bih;
pub mod bvh;
pub mod bvh_rec;
pub mod kdtree;

use crate::mesh::Vertex;

use self::{
    bvh::BoundingVolumeHierarchy,
    bvh_rec::BoundingVolumeHierarchyRec,
    kdtree::KdTree,
    structure::AccelerationStructure,
};

use super::{aabb::BoundingBox, triangle::Triangle};

#[derive(Default)]
pub struct AccelerationStructures {
    bvh: Option<BoundingVolumeHierarchy>,
    bvh_rec: Option<BoundingVolumeHierarchyRec>,
    kdtree: Option<KdTree>,
}

#[derive(Debug, Default)]
pub struct ConstructError;

impl std::error::Error for ConstructError {}

impl std::fmt::Display for ConstructError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Acceleration structure was already constructed.")
    }
}

impl AccelerationStructures {
    pub fn construct(
        &mut self,
        accel: Accel,
        verts: &[Vertex],
        triangles: &[Triangle],
        triangle_bounds: &[BoundingBox],
    ) -> Result<(), ConstructError> {
        match accel {
            Accel::Bvh => {
                if self.bvh.is_some() {
                    return Err(ConstructError::default());
                }
                let _ = self
                    .bvh
                    .insert(BoundingVolumeHierarchy::new(verts, triangles, triangle_bounds));
            }
            Accel::BvhRecursive => {
                if self.bvh_rec.is_some() {
                    return Err(ConstructError::default());
                }
                let _ = self
                    .bvh_rec
                    .insert(BoundingVolumeHierarchyRec::new(verts, triangles, triangle_bounds));
            }
            Accel::KdTree => {
                if self.kdtree.is_some() {
                    return Err(ConstructError::default());
                }
                let _ = self.kdtree.insert(KdTree::new(verts, triangles, triangle_bounds));
            }
        }
        Ok(())
    }

    pub fn get(&self, accel: Accel) -> &dyn AccelerationStructure {
        match accel {
            Accel::Bvh => self.bvh.as_ref().unwrap(),
            Accel::BvhRecursive => self.bvh_rec.as_ref().unwrap(),
            Accel::KdTree => self.kdtree.as_ref().unwrap(),
        }
    }
}

#[derive(Clone, Copy)]
pub enum Accel {
    Bvh,
    BvhRecursive,
    KdTree,
}

impl std::str::FromStr for Accel {
    type Err = ();

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        match input {
            "bvh" => Ok(Accel::Bvh),
            "bvh_recursive" => Ok(Accel::BvhRecursive),
            "kd_tree" => Ok(Accel::KdTree),
            _ => Err(()),
        }
    }
}
