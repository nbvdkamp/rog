use cgmath::{Point3, Vector3};

use crate::{
    mesh::{Instance, Mesh},
    raytracer::{
        aabb::{BoundingBox, Intersects},
        ray::Ray,
    },
};

use super::{
    bvh::BoundingVolumeHierarchy,
    bvh_rec::BoundingVolumeHierarchyRec,
    kdtree::KdTree,
    statistics::{Statistics, StatisticsStore},
    structure::{AccelerationStructure, TraceResult, TraceResultMesh},
    Accel,
};

pub struct TopLevelBVH {
    children: Vec<Box<dyn AccelerationStructure + Send + Sync>>,
    tree_root: Node,
    stats: Statistics,
}

enum Node {
    Inner {
        left: Box<Node>,
        right: Box<Node>,
        bounds: BoundingBox,
    },
    Leaf {
        instance: Box<Instance>,
    },
}

impl Node {
    fn bounds(&self) -> &BoundingBox {
        match self {
            Node::Inner { bounds, .. } => bounds,
            Node::Leaf { instance } => &instance.bounds,
        }
    }
}

impl<'a> TopLevelBVH {
    pub fn new(accel_type: Accel, meshes: &[Mesh], instances: Vec<Instance>) -> Self {
        let mut children: Vec<Box<dyn AccelerationStructure + Send + Sync>> = Vec::new();

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

        let stats = Statistics::new();

        let mut nodes = instances
            .into_iter()
            .map(|instance| {
                stats.count_leaf_node();
                Node::Leaf {
                    instance: Box::new(instance),
                }
            })
            .collect();

        // From Walter et al. 2008: Fast Agglomerative Clustering for Rendering
        let mut a = 0;
        let mut b = find_best_match(a, &nodes);

        while nodes.len() > 1 {
            let c = find_best_match(b, &nodes);

            if a == c {
                // If a is before b in the vec we need to remove b first so we swap them
                if a < b {
                    (a, b) = (b, a);
                }

                let node_a = nodes.remove(a);
                let node_b = nodes.remove(b);
                let bounds = node_a.bounds().union(node_b.bounds());
                stats.count_inner_node();

                let new_node = Node::Inner {
                    left: Box::new(node_a),
                    right: Box::new(node_b),
                    bounds,
                };
                nodes.push(new_node);
                a = nodes.len() - 1;
                b = find_best_match(a, &nodes);
            } else {
                a = b;
                b = c;
            }
        }

        Self {
            children,
            tree_root: nodes.pop().unwrap(),
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

    fn intersect_node(&'a self, node: &'a Node, ray: &Ray, inv_dir: Vector3<f32>, meshes: &[Mesh]) -> TraceResultMesh {
        match node {
            Node::Inner { left, right, .. } => self.inner_intersect(left, right, ray, inv_dir, meshes),
            Node::Leaf { instance } => {
                let i = instance.mesh_index as usize;
                self.stats.count_intersection_test();
                let mesh = &meshes[i];

                let transformed_ray = Ray {
                    origin: Point3::from_homogeneous(instance.inverse_transform * ray.origin.to_homogeneous()),
                    direction: (instance.inverse_transform * ray.direction.extend(0.0)).truncate(),
                };

                let t = self.children[i].intersect(&transformed_ray, &mesh.vertices.positions, &mesh.triangles);

                if let TraceResult::Hit { .. } = t {
                    self.stats.count_intersection_hit();
                }

                t.with_instance(instance)
            }
        }
    }

    fn inner_intersect(
        &'a self,
        left: &'a Node,
        right: &'a Node,
        ray: &Ray,
        inv_dir: Vector3<f32>,
        meshes: &[Mesh],
    ) -> TraceResultMesh {
        self.stats.count_inner_node_traversal();

        let hit_l_box = left.bounds().intersects_ray(ray, &inv_dir);
        let hit_r_box = right.bounds().intersects_ray(ray, &inv_dir);

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
        &'a self,
        first_hit_child: &'a Node,
        second_hit_child: &'a Node,
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

fn find_best_match(i: usize, nodes: &Vec<Node>) -> usize {
    let node = &nodes[i];
    let mut best = 0;
    let mut best_surface_area = f32::MAX;

    for j in 0..nodes.len() {
        if i == j {
            continue;
        }

        let area = node.bounds().union(nodes[j].bounds()).surface_area();

        if area < best_surface_area {
            best = j;
            best_surface_area = area;
        }
    }

    best
}
