use arrayvec::ArrayVec;
use cgmath::Vector3;

use crate::{
    mesh::Mesh,
    raytracer::{
        aabb::{BoundingBox, Intersects},
        axis::Axis,
        ray::Ray,
    },
};

use super::{
    bvh::BoundingVolumeHierarchy,
    bvh_rec::BoundingVolumeHierarchyRec,
    helpers::compute_bounding_box_item_indexed,
    kdtree::KdTree,
    sah::{surface_area_heuristic_bvh, SurfaceAreaHeuristicResultBvh},
    statistics::{Statistics, StatisticsStore},
    structure::{AccelerationStructure, TraceResult, TraceResultMesh},
    Accel,
};

pub struct TopLevelBVH {
    children: Vec<Box<dyn AccelerationStructure + Sync>>,
    tree_root: Option<Box<Node>>,
    stats: Statistics,
}

enum Node {
    Inner {
        left: Option<Box<Node>>,
        right: Option<Box<Node>>,
        bounds: BoundingBox,
    },
    Leaf {
        mesh_indices: ArrayVec<usize, 10>,
        bounds: BoundingBox,
    },
}

impl TopLevelBVH {
    pub fn new(accel_type: Accel, meshes: &[Mesh]) -> Self {
        let mut children: Vec<Box<dyn AccelerationStructure + Sync>> = Vec::new();

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
                    children.push(Box::new(BoundingVolumeHierarchy::new(
                        &mesh.vertices.positions,
                        &mesh.triangles,
                        &triangle_bounds,
                    )));
                }
                Accel::BvhRecursive => {
                    children.push(Box::new(BoundingVolumeHierarchyRec::new(
                        mesh.triangles.len(),
                        &triangle_bounds,
                    )));
                }
                Accel::KdTree => {
                    children.push(Box::new(KdTree::new(
                        &mesh.vertices.positions,
                        &mesh.triangles,
                        &triangle_bounds,
                    )));
                }
            }
        }

        let mesh_bounds = meshes.iter().map(|mesh| mesh.bounds).collect::<Vec<_>>();
        let mut stats = Statistics::new();
        let tree_root = create_node(&mesh_bounds, (0..meshes.len()).collect(), 0, &mut stats);

        Self {
            children,
            tree_root,
            stats,
        }
    }

    pub fn intersect(&self, ray: &Ray, meshes: &[Mesh]) -> TraceResultMesh {
        self.stats.count_ray();

        self.intersect_node(&self.tree_root, ray, 1.0 / ray.direction, meshes)
    }

    pub fn get_statistics(&self) -> StatisticsStore {
        let mut result = StatisticsStore::new();

        for c in &self.children {
            result = result + c.get_statistics();
        }

        return result;
    }

    pub fn get_top_level_statistics(&self) -> StatisticsStore {
        self.stats.get_copy()
    }

    fn intersect_node(
        &self,
        node_opt: &Option<Box<Node>>,
        ray: &Ray,
        inv_dir: Vector3<f32>,
        meshes: &[Mesh],
    ) -> TraceResultMesh {
        match node_opt {
            Some(node) => match node.as_ref() {
                Node::Inner { left, right, .. } => self.inner_intersect(left, right, ray, inv_dir, meshes),
                Node::Leaf { mesh_indices, .. } => {
                    let mut result = TraceResult::Miss;
                    let mut mesh_index = 0;

                    for &i in mesh_indices {
                        self.stats.count_intersection_test();
                        let mesh = &meshes[i];
                        let t = self.children[i].intersect(ray, &mesh.vertices.positions, &mesh.triangles);

                        if let TraceResult::Hit { .. } = t {
                            self.stats.count_intersection_hit();
                        }

                        if t.is_closer_than(&result) {
                            result = t;
                            mesh_index = i;
                        }
                    }

                    result.with_mesh_index(mesh_index)
                }
            },
            None => TraceResultMesh::Miss,
        }
    }

    fn inner_intersect(
        &self,
        left: &Option<Box<Node>>,
        right: &Option<Box<Node>>,
        ray: &Ray,
        inv_dir: Vector3<f32>,
        meshes: &[Mesh],
    ) -> TraceResultMesh {
        self.stats.count_inner_node_traversal();

        let hit_l_box = intersects_bounds(left, ray, inv_dir);
        let hit_r_box = intersects_bounds(right, ray, inv_dir);

        match (hit_l_box, hit_r_box) {
            (Intersects::No, Intersects::No) => TraceResultMesh::Miss,
            (Intersects::Yes { .. }, Intersects::No) => self.intersect_node(left, ray, inv_dir, meshes),
            (Intersects::No, Intersects::Yes { .. }) => self.intersect_node(right, ray, inv_dir, meshes),
            (Intersects::Yes { distance: l_distance }, Intersects::Yes { distance: r_distance }) => {
                if l_distance < r_distance {
                    self.intersect_both_children_hit(left, right, r_distance, ray, inv_dir, meshes)
                } else {
                    self.intersect_both_children_hit(right, left, l_distance, ray, inv_dir, meshes)
                }
            }
        }
    }

    fn intersect_both_children_hit(
        &self,
        first_hit_child: &Option<Box<Node>>,
        second_hit_child: &Option<Box<Node>>,
        dist_to_second_box: f32,
        ray: &Ray,
        inv_dir: Vector3<f32>,
        meshes: &[Mesh],
    ) -> TraceResultMesh {
        let first_result = self.intersect_node(first_hit_child, ray, inv_dir, meshes);

        let TraceResultMesh::Hit { t: t_first, .. } = first_result else {
            return self.intersect_node(second_hit_child, ray, inv_dir, meshes)
        };

        if t_first < dist_to_second_box {
            first_result
        } else {
            let second_result = self.intersect_node(second_hit_child, ray, inv_dir, meshes);

            if let TraceResultMesh::Hit { t: t_second, .. } = second_result {
                if t_second < t_first {
                    second_result
                } else {
                    first_result
                }
            } else {
                first_result
            }
        }
    }
}

fn intersects_bounds(node_opt: &Option<Box<Node>>, ray: &Ray, inv_dir: Vector3<f32>) -> Intersects {
    match node_opt {
        Some(node) => {
            let node = node.as_ref();
            match node {
                Node::Inner { bounds, .. } => bounds.intersects_ray(ray, &inv_dir),
                Node::Leaf { bounds, .. } => bounds.intersects_ray(ray, &inv_dir),
            }
        }
        None => Intersects::No,
    }
}

fn create_node(
    item_bounds: &[BoundingBox],
    item_indices: Vec<usize>,
    depth: usize,
    stats: &mut Statistics,
) -> Option<Box<Node>> {
    if item_indices.is_empty() {
        return None;
    }

    stats.count_max_depth(depth);

    let bounds = compute_bounding_box_item_indexed(item_bounds, &item_indices);
    let axes_to_search = [Axis::X, Axis::Y, Axis::Z];

    match surface_area_heuristic_bvh(&item_bounds, item_indices, bounds, &axes_to_search, 1.1) {
        SurfaceAreaHeuristicResultBvh::MakeLeaf { indices } => {
            stats.count_leaf_node();

            Some(Box::new(Node::Leaf {
                mesh_indices: indices.into_iter().collect(),
                bounds,
            }))
        }
        SurfaceAreaHeuristicResultBvh::MakeInner {
            left_indices,
            right_indices,
        } => {
            stats.count_inner_node();

            let left = create_node(item_bounds, left_indices, depth + 1, stats);
            let right = create_node(item_bounds, right_indices, depth + 1, stats);

            Some(Box::new(Node::Inner { left, right, bounds }))
        }
    }
}
