use crate::raytracer::{aabb::BoundingBox, axis::Axis};

const BUCKET_COUNT: usize = 12;
const POTENTIAL_SPLIT_COUNT: usize = BUCKET_COUNT - 1;
const RELATIVE_TRAVERSAL_COST: f32 = 1.2;
const MAX_TRIS_IN_LEAF: usize = 255;

pub enum SurfaceAreaHeuristicResult {
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

pub fn surface_area_heuristic(
    triangle_bounds: &[BoundingBox],
    triangle_indices: Vec<usize>,
    split_axis: Axis,
    bounds: BoundingBox,
) -> SurfaceAreaHeuristicResult {
    let axis_index = split_axis.index();

    let mut centroid_bounds = BoundingBox::new();

    for index in &triangle_indices {
        centroid_bounds.add(triangle_bounds[*index].center());
    }

    let centroid_bounds_extent = centroid_bounds.max - centroid_bounds.min;

    let mut buckets = [Bucket {
        count: 0,
        bounds: BoundingBox::new(),
    }; BUCKET_COUNT];

    let bucket_index = |center| {
        let x = (center - centroid_bounds.min[axis_index]) / centroid_bounds_extent[axis_index];

        ((BUCKET_COUNT as f32 * x) as usize).min(BUCKET_COUNT - 1)
    };

    for index in &triangle_indices {
        let bounds = triangle_bounds[*index];

        let center = bounds.center()[axis_index];
        let bucket = &mut buckets[bucket_index(center)];
        bucket.count += 1;
        bucket.bounds = bucket.bounds.union(bounds);
    }

    let mut costs = [0.0; POTENTIAL_SPLIT_COUNT];

    for i in 0..POTENTIAL_SPLIT_COUNT {
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

    let should_make_leaf = triangle_indices.len() < MAX_TRIS_IN_LEAF && min_cost > triangle_indices.len() as f32;

    if should_make_leaf {
        return SurfaceAreaHeuristicResult::MakeLeaf {
            indices: triangle_indices,
        };
    }

    let (left_indices, right_indices) = triangle_indices
        .into_iter()
        .partition(|i| bucket_index(triangle_bounds[*i].center()[axis_index]) <= min_index);

    SurfaceAreaHeuristicResult::MakeInner {
        left_indices,
        right_indices,
    }
}
