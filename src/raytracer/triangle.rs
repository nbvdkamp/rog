use super::aabb::BoundingBox;

pub struct Triangle {
    pub index1: u32,
    pub index2: u32,
    pub index3: u32,
    pub material_index: u32,
    pub bounds: BoundingBox,
}