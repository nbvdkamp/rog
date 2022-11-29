use crate::raytracer::{aabb::BoundingBox, axis::Axis};

const BUCKET_COUNT: usize = 12;
const POTENTIAL_SPLIT_COUNT: usize = BUCKET_COUNT - 1;
const RELATIVE_TRAVERSAL_COST: f32 = 1.2;
const MAX_TRIS_IN_LEAF: usize = 255;

pub enum SurfaceAreaHeuristicResultBvh {
    MakeLeaf {
        indices: Vec<usize>,
    },
    MakeInner {
        left_indices: Vec<usize>,
        right_indices: Vec<usize>,
    },
}

#[derive(Clone, Copy)]
struct Bucket {
    count: u32,
    bounds: BoundingBox,
}

pub fn surface_area_heuristic_bvh(
    item_bounds: &[BoundingBox],
    item_indices: Vec<usize>,
    bounds: BoundingBox,
    axes_to_search: &[Axis],
    relative_traversal_cost: f32,
) -> SurfaceAreaHeuristicResultBvh {
    let mut centroid_bounds = BoundingBox::new();

    for index in &item_indices {
        centroid_bounds.add(item_bounds[*index].center());
    }

    let centroid_bounds_extent = centroid_bounds.max - centroid_bounds.min;

    let bucket_index = |center, axis_index: usize| {
        let x = (center - centroid_bounds.min[axis_index]) / centroid_bounds_extent[axis_index];

        ((BUCKET_COUNT as f32 * x) as usize).min(BUCKET_COUNT - 1)
    };

    let mut min_cost = f32::MAX;
    let mut min_index = 0;
    let mut min_axis = Axis::X;

    for &split_axis in axes_to_search {
        let axis_index = split_axis.index();

        let mut buckets = [Bucket {
            count: 0,
            bounds: BoundingBox::new(),
        }; BUCKET_COUNT];

        for index in &item_indices {
            let bounds = item_bounds[*index];

            let center = bounds.center()[axis_index];
            let bucket = &mut buckets[bucket_index(center, axis_index)];
            bucket.count += 1;
            bucket.bounds = bucket.bounds.union(&bounds);
        }

        let mut costs = [0.0; POTENTIAL_SPLIT_COUNT];

        #[allow(clippy::needless_range_loop)]
        for i in 0..POTENTIAL_SPLIT_COUNT {
            let mut b0 = BoundingBox::new();
            let mut b1 = BoundingBox::new();

            let mut count0 = 0;
            let mut count1 = 0;

            for j in 0..=i {
                b0 = b0.union(&buckets[j].bounds);
                count0 += buckets[j].count;
            }
            for j in i + 1..BUCKET_COUNT {
                b1 = b1.union(&buckets[j].bounds);
                count1 += buckets[j].count;
            }

            let approx_children_cost =
                (count0 as f32 * b0.surface_area() + count1 as f32 * b1.surface_area()) / bounds.surface_area();
            costs[i] = relative_traversal_cost + approx_children_cost;
        }

        for (i, cost) in costs.into_iter().enumerate() {
            if cost < min_cost {
                min_cost = cost;
                min_index = i;
                min_axis = split_axis;
            }
        }
    }

    let should_make_leaf = item_indices.len() < MAX_TRIS_IN_LEAF && min_cost > item_indices.len() as f32;

    if should_make_leaf {
        return SurfaceAreaHeuristicResultBvh::MakeLeaf { indices: item_indices };
    }

    let axis_index = min_axis.index();

    let (left_indices, right_indices) = item_indices
        .into_iter()
        .partition(|i| bucket_index(item_bounds[*i].center()[axis_index], axis_index) <= min_index);

    SurfaceAreaHeuristicResultBvh::MakeInner {
        left_indices,
        right_indices,
    }
}

pub enum SurfaceAreaHeuristicResultKdTree {
    MakeLeaf {
        indices: Vec<usize>,
    },
    MakeInner {
        split_axis: Axis,
        split_position: f32,
        left_indices: Vec<usize>,
        right_indices: Vec<usize>,
    },
}

pub fn surface_area_heuristic_kd_tree(
    triangle_bounds: &[BoundingBox],
    triangle_indices: Vec<usize>,
    bounds: BoundingBox,
) -> SurfaceAreaHeuristicResultKdTree {
    let mut min_cost = f32::MAX;
    let mut min_index = 0;
    let mut min_axis = Axis::X;

    let split_pos = |i, axis_index| {
        bounds.min[axis_index]
            + ((i + 1) as f32 / BUCKET_COUNT as f32) * (bounds.max[axis_index] - bounds.min[axis_index])
    };

    for split_axis in [Axis::X, Axis::Y, Axis::Z] {
        let axis_index = split_axis.index();
        let mut costs = [0.0; POTENTIAL_SPLIT_COUNT];

        #[allow(clippy::needless_range_loop)]
        for i in 0..POTENTIAL_SPLIT_COUNT {
            let split_pos = split_pos(i, axis_index);
            let mut b0 = bounds;
            b0.set_max(split_axis, split_pos);
            let mut b1 = bounds;
            b1.set_min(split_axis, split_pos);

            let mut count0 = 0;
            let mut count1 = 0;

            for index in &triangle_indices {
                let bounds = triangle_bounds[*index];

                if bounds.min[axis_index] <= split_pos {
                    count0 += 1;
                }
                if bounds.max[axis_index] > split_pos {
                    count1 += 1;
                }
            }

            let approx_children_cost =
                (count0 as f32 * b0.surface_area() + count1 as f32 * b1.surface_area()) / bounds.surface_area();
            costs[i] = RELATIVE_TRAVERSAL_COST + approx_children_cost;
        }

        for (i, cost) in costs.into_iter().enumerate() {
            if cost < min_cost {
                min_cost = cost;
                min_index = i;
                min_axis = split_axis;
            }
        }
    }

    let axis_index = min_axis.index();
    let split_position = split_pos(min_index, axis_index);

    let should_make_leaf = triangle_indices.len() < MAX_TRIS_IN_LEAF && min_cost > triangle_indices.len() as f32;

    if should_make_leaf {
        return SurfaceAreaHeuristicResultKdTree::MakeLeaf {
            indices: triangle_indices,
        };
    }

    let left_indices = triangle_indices
        .iter()
        .filter(|&&i| triangle_bounds[i].min[axis_index] <= split_position)
        .copied()
        .collect();

    let right_indices = triangle_indices
        .into_iter()
        .filter(|&i| triangle_bounds[i].max[axis_index] > split_position)
        .collect();

    SurfaceAreaHeuristicResultKdTree::MakeInner {
        split_axis: min_axis,
        split_position,
        left_indices,
        right_indices,
    }
}
