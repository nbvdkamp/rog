pub mod helpers;
pub mod sah;
pub mod statistics;
pub mod structure;

pub mod top_level;

// pub mod bih;
pub mod bvh;
pub mod bvh_rec;
pub mod kdtree;

use crate::mesh::{Instance, Mesh};

use self::top_level::TopLevelBVH;

#[derive(Default)]
pub struct AccelerationStructures {
    bvh: Option<TopLevelBVH>,
    bvh_rec: Option<TopLevelBVH>,
    kdtree: Option<TopLevelBVH>,
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
    pub fn construct(&mut self, accel: Accel, meshes: &[Mesh], instances: Vec<Instance>) -> Result<(), ConstructError> {
        match accel {
            Accel::Bvh => {
                if self.bvh.is_some() {
                    return Err(ConstructError);
                }
                let _ = self.bvh.insert(TopLevelBVH::new(accel, meshes, instances));
            }
            Accel::BvhRecursive => {
                if self.bvh_rec.is_some() {
                    return Err(ConstructError);
                }
                let _ = self.bvh_rec.insert(TopLevelBVH::new(accel, meshes, instances));
            }
            Accel::KdTree => {
                if self.kdtree.is_some() {
                    return Err(ConstructError);
                }
                let _ = self.kdtree.insert(TopLevelBVH::new(accel, meshes, instances));
            }
        }
        Ok(())
    }

    pub fn get(&self, accel: Accel) -> &TopLevelBVH {
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

impl Accel {
    pub fn name(&self) -> &str {
        match self {
            Accel::Bvh => "BVH (iterative)",
            Accel::BvhRecursive => "BVH (recursive)",
            Accel::KdTree => "KD Tree",
        }
    }
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
