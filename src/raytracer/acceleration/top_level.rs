use crate::{
    mesh::Mesh,
    raytracer::{aabb::BoundingBox, ray::Ray},
};

use super::{
    bvh::BoundingVolumeHierarchy,
    bvh_rec::BoundingVolumeHierarchyRec,
    kdtree::KdTree,
    statistics::StatisticsStore,
    structure::{AccelerationStructure, TraceResult, TraceResultMesh},
    Accel,
};

pub struct TopLevelBVH {
    children: Vec<Box<dyn AccelerationStructure + Sync>>,
}

impl TopLevelBVH {
    pub fn new(accel_type: Accel, meshes: &[Mesh]) -> Self {
        let mut result = Self { children: Vec::new() };

        for mesh in meshes {
            let triangle_bounds = mesh
                .triangles
                .iter()
                .map(|tri| {
                    let mut b = BoundingBox::new();
                    b.add(mesh.vertices.positions[tri.indices[0] as usize]);
                    b.add(mesh.vertices.positions[tri.indices[1] as usize]);
                    b.add(mesh.vertices.positions[tri.indices[2] as usize]);
                    b
                })
                .collect::<Vec<_>>();

            match accel_type {
                Accel::Bvh => {
                    result.children.push(Box::new(BoundingVolumeHierarchy::new(
                        &mesh.vertices.positions,
                        &mesh.triangles,
                        &triangle_bounds,
                    )));
                }
                Accel::BvhRecursive => {
                    result.children.push(Box::new(BoundingVolumeHierarchyRec::new(
                        mesh.triangles.len(),
                        &triangle_bounds,
                    )));
                }
                Accel::KdTree => {
                    result.children.push(Box::new(KdTree::new(
                        &mesh.vertices.positions,
                        &mesh.triangles,
                        &triangle_bounds,
                    )));
                }
            }
        }

        result
    }

    pub fn intersect(&self, ray: &Ray, meshes: &[Mesh]) -> TraceResultMesh {
        let mut result = TraceResult::Miss;
        let mut mesh_index = 0;
        assert_eq!(meshes.len(), self.children.len());

        for i in 0..meshes.len() {
            let mesh = &meshes[i];
            let t = self.children[i].intersect(ray, &mesh.vertices.positions, &mesh.triangles);

            if t.is_closer_than(&result) {
                result = t;
                mesh_index = i;
            }
        }

        result.with_mesh_index(mesh_index)
    }

    pub fn get_statistics(&self) -> StatisticsStore {
        let mut result = StatisticsStore::new();

        for c in &self.children {
            result = result + c.get_statistics();
        }

        return result;
    }
}
