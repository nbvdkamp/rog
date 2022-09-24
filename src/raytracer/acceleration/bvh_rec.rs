use crate::{
    mesh::Vertex,
    raytracer::{axis::Axis, triangle::Triangle, Ray},
};

use cgmath::Vector3;

use super::{
    super::aabb::BoundingBox,
    helpers::{compute_bounding_box_triangle_indexed, intersect_triangles_indexed},
    statistics::{Statistics, StatisticsStore},
    structure::{AccelerationStructure, TraceResult},
};

pub struct BoundingVolumeHierarchyRec {
    root: Option<Box<Node>>,
    stats: Statistics,
}

enum Node {
    Leaf {
        triangle_indices: Vec<usize>,
        bounds: BoundingBox,
    },
    Inner {
        left_child: Option<Box<Node>>,
        right_child: Option<Box<Node>>,
        bounds: BoundingBox,
    },
}

impl AccelerationStructure for BoundingVolumeHierarchyRec {
    #[allow(clippy::only_used_in_recursion)]
    fn intersect(&self, ray: &Ray, verts: &[Vertex], triangles: &[Triangle]) -> TraceResult {
        self.stats.count_ray();

        let inv_dir = 1.0 / ray.direction;

        if intersects_bounds(&self.root, ray, inv_dir) {
            self.intersect(&self.root, ray, inv_dir, verts, triangles)
        } else {
            TraceResult::Miss
        }
    }

    fn get_name(&self) -> &str {
        "BVH (recursive)"
    }

    fn get_statistics(&self) -> StatisticsStore {
        self.stats.get_copy()
    }
}

impl BoundingVolumeHierarchyRec {
    pub fn new(verts: &[Vertex], triangles: &[Triangle], triangle_bounds: &[BoundingBox]) -> Self {
        let mut item_indices = Vec::new();
        let mut stats = Statistics::new();

        for i in 0..triangles.len() {
            item_indices.push(i);
        }

        BoundingVolumeHierarchyRec {
            root: create_node(verts, triangles, triangle_bounds, item_indices, 0, Axis::X, &mut stats),
            stats,
        }
    }

    fn intersect(
        &self,
        node_opt: &Option<Box<Node>>,
        ray: &Ray,
        inv_dir: Vector3<f32>,
        verts: &[Vertex],
        triangles: &[Triangle],
    ) -> TraceResult {
        match node_opt {
            Some(node) => {
                let node = node.as_ref();
                match node {
                    Node::Inner {
                        left_child,
                        right_child,
                        ..
                    } => self.inner_intersect(left_child, right_child, ray, inv_dir, verts, triangles),
                    Node::Leaf { triangle_indices, .. } => {
                        intersect_triangles_indexed(triangle_indices, ray, verts, triangles, &self.stats)
                    }
                }
            }
            None => TraceResult::Miss,
        }
    }

    fn inner_intersect(
        &self,
        left: &Option<Box<Node>>,
        right: &Option<Box<Node>>,
        ray: &Ray,
        inv_dir: Vector3<f32>,
        verts: &[Vertex],
        triangles: &[Triangle],
    ) -> TraceResult {
        self.stats.count_inner_node_traversal();

        let hit_l_box = intersects_bounds(left, ray, inv_dir);
        let hit_r_box = intersects_bounds(right, ray, inv_dir);

        if !hit_l_box && !hit_r_box {
            return TraceResult::Miss;
        } else if hit_l_box && !hit_r_box {
            return self.intersect(left, ray, inv_dir, verts, triangles);
        } else if !hit_l_box && hit_r_box {
            return self.intersect(right, ray, inv_dir, verts, triangles);
        }

        // Both children are intersected

        let dist_to_left_box = intersects_bounds_distance(left, ray, inv_dir);
        let dist_to_right_box = intersects_bounds_distance(right, ray, inv_dir);

        if dist_to_left_box < dist_to_right_box {
            self.intersect_both_children_hit(left, right, dist_to_right_box, ray, inv_dir, verts, triangles)
        } else {
            self.intersect_both_children_hit(right, left, dist_to_left_box, ray, inv_dir, verts, triangles)
        }
    }

    fn intersect_both_children_hit(
        &self,
        first_hit_child: &Option<Box<Node>>,
        second_hit_child: &Option<Box<Node>>,
        dist_to_second_box: f32,
        ray: &Ray,
        inv_dir: Vector3<f32>,
        verts: &[Vertex],
        triangles: &[Triangle],
    ) -> TraceResult {
        let first_result = self.intersect(first_hit_child, ray, inv_dir, verts, triangles);

        if let TraceResult::Hit { t: t_first, .. } = first_result {
            if t_first < dist_to_second_box {
                first_result
            } else {
                let second_result = self.intersect(second_hit_child, ray, inv_dir, verts, triangles);

                if let TraceResult::Hit { t: t_second, .. } = second_result {
                    if t_second < t_first {
                        second_result
                    } else {
                        first_result
                    }
                } else {
                    first_result
                }
            }
        } else {
            self.intersect(second_hit_child, ray, inv_dir, verts, triangles)
        }
    }
}

fn intersects_bounds(node_opt: &Option<Box<Node>>, ray: &Ray, inv_dir: Vector3<f32>) -> bool {
    match node_opt {
        Some(node) => {
            let node = node.as_ref();
            match node {
                Node::Inner { bounds, .. } => bounds.intersects_ray(ray, &inv_dir),
                Node::Leaf { bounds, .. } => bounds.intersects_ray(ray, &inv_dir),
            }
        }
        None => false,
    }
}

fn intersects_bounds_distance(node_opt: &Option<Box<Node>>, ray: &Ray, inv_dir: Vector3<f32>) -> f32 {
    match node_opt {
        Some(node) => {
            let node = node.as_ref();
            match node {
                Node::Inner { bounds, .. } => bounds.t_distance_from_ray(ray, &inv_dir),
                Node::Leaf { bounds, .. } => bounds.t_distance_from_ray(ray, &inv_dir),
            }
        }
        None => {
            unreachable!()
        }
    }
}

fn create_node(
    verts: &[Vertex],
    triangles: &[Triangle],
    triangle_bounds: &[BoundingBox],
    triangle_indices: Vec<usize>,
    depth: usize,
    split_axis: Axis,
    stats: &mut Statistics,
) -> Option<Box<Node>> {
    if triangle_indices.is_empty() {
        return None;
    }

    stats.count_max_depth(depth);

    let bounds = compute_bounding_box_triangle_indexed(triangle_bounds, &triangle_indices);

    let axis_index = split_axis.index();

    let mut centroid_bounds = BoundingBox::new();

    triangle_indices.iter().for_each(|index| {
        centroid_bounds.add(triangle_bounds[*index].center());
    });

    const BUCKET_COUNT: usize = 12;

    #[derive(Clone, Copy)]
    struct Bucket {
        count: u32,
        bounds: BoundingBox,
    }

    let mut buckets = [Bucket {
        count: 0,
        bounds: BoundingBox::new(),
    }; BUCKET_COUNT];

    let bucket_index = |center| {
        let x = (center - centroid_bounds.min[axis_index])
            / (centroid_bounds.max[axis_index] - centroid_bounds.min[axis_index]);
        ((BUCKET_COUNT as f32 * x) as usize).min(BUCKET_COUNT - 1)
    };

    triangle_indices.iter().for_each(|index| {
        let bounds = triangle_bounds[*index];

        let center = bounds.center()[axis_index];
        let bucket = &mut buckets[bucket_index(center)];
        bucket.count += 1;
        bucket.bounds = bucket.bounds.union(bounds);
    });

    let mut costs = [0.0; BUCKET_COUNT - 1];

    for i in 0..BUCKET_COUNT - 1 {
        let mut b0 = BoundingBox::new();
        let mut b1 = BoundingBox::new();

        let mut count0 = 0;
        let mut count1 = 0;

        for j in 0..=i {
            b0 = b0.union(buckets[j].bounds);
            count0 += buckets[j].count;
        }
        for j in i + 1..BUCKET_COUNT {
            b1 = b1.union(buckets[j].bounds);
            count1 += buckets[j].count;
        }

        const RELATIVE_TRAVERSAL_COST: f32 = 1.2;
        let approx_children_cost =
            (count0 as f32 * b0.surface_area() + count1 as f32 * b1.surface_area()) / bounds.surface_area();
        costs[i] = RELATIVE_TRAVERSAL_COST + approx_children_cost;
    }

    let mut min_cost = costs[0];
    let mut min_index = 0;

    for i in 1..BUCKET_COUNT - 1 {
        if costs[i] < min_cost {
            min_cost = costs[i];
            min_index = i;
        }
    }

    const MAX_TRIS_IN_LEAF: usize = 255;

    let should_make_inner = triangle_indices.len() > MAX_TRIS_IN_LEAF || min_cost < triangle_indices.len() as f32;

    if !should_make_inner {
        stats.count_leaf_node();

        return Some(Box::new(Node::Leaf {
            triangle_indices,
            bounds,
        }));
    };

    let (left_indices, right_indices) = triangle_indices
        .into_iter()
        .partition(|i| bucket_index(triangle_bounds[*i].center()[axis_index]) <= min_index);

    let left = create_node(
        verts,
        triangles,
        triangle_bounds,
        left_indices,
        depth + 1,
        split_axis.next(),
        stats,
    );
    let right = create_node(
        verts,
        triangles,
        triangle_bounds,
        right_indices,
        depth + 1,
        split_axis.next(),
        stats,
    );

    stats.count_inner_node();

    Some(Box::new(Node::Inner {
        left_child: left,
        right_child: right,
        bounds,
    }))
}
