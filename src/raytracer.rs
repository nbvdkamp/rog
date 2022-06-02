use rand::{thread_rng, Rng};
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use cgmath::{point3, vec2, vec3, EuclideanSpace, InnerSpace, Point3, Vector2, Vector3, Vector4};
use crossbeam::{
    deque::{Injector, Steal},
    scope,
};

mod aabb;
mod acceleration;
mod axis;
mod bsdf;
pub mod geometry;
mod ray;
mod shadingframe;
mod triangle;

use crate::{
    camera::PerspectiveCamera,
    environment::Environment,
    light::Light,
    material::Material,
    mesh::Vertex,
    scene::Scene,
    spectrum::Spectrumf32,
    texture::{Format, Texture},
};
use ray::{IntersectionResult, Ray};
use triangle::Triangle;

use aabb::BoundingBox;
use acceleration::{
    bvh::BoundingVolumeHierarchy,
    bvh_rec::BoundingVolumeHierarchyRec,
    kdtree::KdTree,
    structure::{AccelerationStructure, TraceResult},
};
use bsdf::{mis2, Evaluation, Sample};
use shadingframe::ShadingFrame;

use geometry::ensure_valid_reflection;

pub struct Raytracer {
    verts: Vec<Vertex>,
    triangles: Vec<Triangle>,
    materials: Vec<Material>,
    lights: Vec<Light>,
    textures: Vec<Texture>,
    environment: Environment,
    pub camera: PerspectiveCamera,
    max_depth: usize,
    pub accel_structures: Vec<Box<dyn AccelerationStructure + Sync>>,
}

impl Raytracer {
    pub fn new(scene: &Scene) -> Self {
        let mut verts = Vec::new();
        let mut triangles = Vec::new();
        let mut materials = Vec::new();

        // TODO: Fix borrowing to prevent having to clone everything
        for mesh in &scene.meshes {
            let start_index = verts.len() as u32;
            let material_index = materials.len() as u32;

            for v in &mesh.vertices {
                verts.push(v.clone());
            }

            materials.push(mesh.material.clone());

            for i in (0..mesh.indices.len()).step_by(3) {
                let index1 = mesh.indices[i] + start_index;
                let index2 = mesh.indices[i + 1] + start_index;
                let index3 = mesh.indices[i + 2] + start_index;

                let mut bounds = BoundingBox::new();
                bounds.add(&verts[index1 as usize].position);
                bounds.add(&verts[index2 as usize].position);
                bounds.add(&verts[index3 as usize].position);

                triangles.push(Triangle {
                    index1,
                    index2,
                    index3,
                    material_index,
                    bounds,
                });
            }
        }

        let mut result = Raytracer {
            verts,
            triangles,
            materials,
            lights: scene.lights.clone(),
            textures: scene.textures.clone(), // FIXME: This wastes a lot of memory
            environment: scene.environment.clone(),
            camera: scene.camera,
            max_depth: 10,
            accel_structures: Vec::new(),
        };

        // result.accel_structures.push(Box::new(BoundingIntervalHierarchy::new(&result.verts, &result.triangles)));
        result
            .accel_structures
            .push(Box::new(BoundingVolumeHierarchy::new(&result.verts, &result.triangles)));
        result.accel_structures.push(Box::new(BoundingVolumeHierarchyRec::new(
            &result.verts,
            &result.triangles,
        )));
        result
            .accel_structures
            .push(Box::new(KdTree::new(&result.verts, &result.triangles)));

        result
    }

    pub fn render(&self, image_size: Vector2<usize>, samples: usize, accel_index: usize) -> (Vec<Spectrumf32>, f32) {
        let aspect_ratio = image_size.x as f32 / image_size.y as f32;
        let fov_factor = (self.camera.y_fov / 2.).tan();

        let start = Instant::now();

        let cam_model = self.camera.model;
        let cam_pos4 = cam_model * Vector4::new(0., 0., 0., 1.);
        let camera_pos = Point3::from_homogeneous(cam_pos4);
        let buffer = vec![Spectrumf32::constant(0.0); image_size.x * image_size.y];
        let buffer = Arc::new(Mutex::new(buffer));

        let thread_count = usize::max(num_cpus::get() - 2, 1);
        let tile_size = 100;

        struct Tile {
            start: Vector2<usize>,
            end: Vector2<usize>,
        }

        let tiles = &{
            let tiles: Injector<Tile> = Injector::new();

            for x_start in (0..image_size.x).step_by(tile_size) {
                let x_end = (x_start + tile_size).min(image_size.x);

                for y_start in (0..image_size.y).step_by(tile_size) {
                    let y_end = (y_start + tile_size).min(image_size.y);

                    tiles.push(Tile {
                        start: vec2(x_start, y_start),
                        end: vec2(x_end, y_end),
                    });
                }
            }

            tiles
        };

        scope(|s| {
            for _ in 0..thread_count {
                let buffer = Arc::clone(&buffer);

                s.spawn(move |_| {
                    'work: loop {
                        let tile = loop {
                            match tiles.steal() {
                                Steal::Success(tile) => break tile,
                                Steal::Empty => break 'work,
                                Steal::Retry => {}
                            }
                        };

                        for y in tile.start.y..tile.end.y {
                            for x in tile.start.x..tile.end.x {
                                let mut color = Spectrumf32::constant(0.0);

                                for sample in 0..samples {
                                    let mut rng = rand::thread_rng();

                                    let mut tent_sample = || {
                                        let r = 2.0 * rng.gen::<f32>();

                                        if r < 1.0 {
                                            r.sqrt() - 1.0
                                        } else {
                                            1.0 - (2.0 - r).sqrt()
                                        }
                                    };

                                    let mut offset = vec2(0.5, 0.5);

                                    if sample > 0 {
                                        offset += vec2(tent_sample(), tent_sample());
                                    }

                                    let screen = self.pixel_to_screen(
                                        Vector2::new(x, y),
                                        offset,
                                        image_size,
                                        aspect_ratio,
                                        fov_factor,
                                    );

                                    // Using w = 0 because this is a direction vector
                                    let dir4 = cam_model * Vector4::new(screen.x, screen.y, -1., 0.).normalize();
                                    let ray = Ray {
                                        origin: camera_pos,
                                        direction: dir4.truncate().normalize(),
                                    };

                                    color += self.radiance(ray, accel_index);
                                }

                                color /= samples as f32;

                                let mut buffer = buffer.lock().unwrap();
                                buffer[image_size.x * y + x] = color;
                            }
                        }
                    }
                });
            }
        })
        .unwrap();

        // Errors when the lock has multiple owners but the scope should guarantee that never happens
        let lock = Arc::try_unwrap(buffer).ok().unwrap();
        let buffer = lock.into_inner().expect("Cannot unlock buffer mutex");

        (buffer, start.elapsed().as_secs_f32())
    }

    fn pixel_to_screen(
        &self,
        pixel: Vector2<usize>,
        offset: Vector2<f32>,
        image_size: Vector2<usize>,
        aspect_ratio: f32,
        fov_factor: f32,
    ) -> Vector2<f32> {
        let normalized_x = (pixel.x as f32 + offset.x) / image_size.x as f32;
        let normalized_y = (pixel.y as f32 + offset.y) / image_size.y as f32;
        let x = (2. * normalized_x - 1.) * fov_factor * aspect_ratio;
        let y = (1. - 2. * normalized_y) * fov_factor;
        Vector2::new(x, y)
    }

    pub fn get_num_tris(&self) -> usize {
        self.triangles.len()
    }

    fn radiance(&self, ray: Ray, accel_index: usize) -> Spectrumf32 {
        let mut result = Spectrumf32::constant(0.0);
        let mut path_weight = Spectrumf32::constant(1.0);
        let mut ray = ray;

        let num_lights = self.lights.len();
        let mut depth = 0;

        while depth < self.max_depth {
            if let TraceResult::Hit {
                triangle_index, u, v, ..
            } = self.trace(&ray, accel_index)
            {
                let triangle = &self.triangles[triangle_index as usize];
                let material = &self.materials[triangle.material_index as usize];

                let verts = [
                    &self.verts[triangle.index1 as usize],
                    &self.verts[triangle.index2 as usize],
                    &self.verts[triangle.index3 as usize],
                ];

                let w = 1.0 - u - v;
                let hit_pos = w * verts[0].position + u * verts[1].position.to_vec() + v * verts[2].position.to_vec();

                let has_texture_coords = verts[0].tex_coord.is_some();

                let texture_coords = if has_texture_coords {
                    w * verts[0].tex_coord.unwrap() + u * verts[1].tex_coord.unwrap() + v * verts[2].tex_coord.unwrap()
                } else {
                    vec2(0.0, 0.0)
                };

                let mat_sample = material.sample(texture_coords, &self.textures);

                if thread_rng().gen::<f32>() > mat_sample.alpha {
                    // Offset to the back of the triangle
                    ray.origin = hit_pos + ray.direction * 0.0002;

                    // Don't count alpha hits for max bounces
                    continue;
                }

                let edge1 = verts[0].position - verts[1].position;
                let edge2 = verts[0].position - verts[2].position;
                let mut geom_normal = edge1.cross(edge2).normalize();

                //TODO: Determine and pass if in or outside material

                // Interpolate the vertex normals
                let mut normal = (w * verts[0].normal + u * verts[1].normal + v * verts[2].normal).normalize();
                let tangent = (w * verts[0].tangent + u * verts[1].tangent + v * verts[2].tangent).normalize();

                // Flip the computed geometric normal to the same side as the interpolated vertex normal.
                if geom_normal.dot(normal) < 0.0 {
                    geom_normal = -geom_normal;
                }

                let backfacing = geom_normal.dot(-ray.direction) < 0.0;

                if backfacing {
                    normal = -normal;
                }

                let offset_hit_pos = Raytracer::offset_hit_pos(hit_pos, geom_normal);


                let frame = {
                    let f = ShadingFrame::new_with_tangent(normal, tangent);

                    if let Some(shading_normal) = mat_sample.shading_normal {
                        let normal = f.to_global(shading_normal).normalize();
                        let normal = ensure_valid_reflection(geom_normal, -ray.direction, normal);
                        ShadingFrame::new(normal)
                    } else {
                        f
                    }
                };

                let local_incident = frame.to_local(-ray.direction);

                // Next event estimation (directly sampling lights)
                if num_lights > 0 {
                    let light = &self.lights[rand::thread_rng().gen_range(0..num_lights)];
                    let light_sample = light.sample(offset_hit_pos);

                    if light_sample.distance <= light.range {
                        let shadow_ray = Ray {
                            origin: offset_hit_pos,
                            direction: light_sample.direction,
                        };

                        let shadowed = self.cast_shadow_ray(shadow_ray, light_sample.distance, accel_index);

                        if !shadowed {
                            let local_outgoing = frame.to_local(light_sample.direction);
                            let eval = bsdf::eval(&mat_sample, local_incident, local_outgoing);

                            if let Evaluation::Evaluation { brdf, pdf } = eval {
                                let light_pick_prob = 1.0 / num_lights as f32;
                                let light_pdf = light_pick_prob * light_sample.pdf;

                                let mis_weight = if light_sample.use_mis {
                                    mis2(light_pdf, pdf)
                                } else {
                                    1.0
                                };

                                result += path_weight
                                    // * mat_sample.base_color_spectrum
                                    * mis_weight
                                    * light_sample.intensity
                                    * brdf
                                    / light_pdf
                                    * light.spectrum;
                            }
                        }
                    }
                }
                break; // no indirect for now

                let sample = bsdf::sample(&mat_sample, local_incident);

                let (local_bounce_dir, brdf, pdf) = match sample {
                    Sample::Sample { outgoing, brdf, pdf } => (outgoing, brdf, pdf),
                    Sample::Null => break,
                };

                path_weight *= brdf / pdf * mat_sample.base_color_spectrum;

                let continue_prob = path_weight.max_value().max(1.0);

                if rand::thread_rng().gen::<f32>() < continue_prob {
                    path_weight /= continue_prob;
                } else {
                    break;
                }

                ray = Ray {
                    origin: offset_hit_pos,
                    direction: frame.to_global(local_bounce_dir),
                };
            } else {
                result += path_weight * self.environment.sample(ray.direction);
                break;
            }

            depth += 1;
        }

        result
    }

    /// Checks if distance to the nearest obstructing triangle is less than the distance to the light
    /// Handles alpha by checking if R ~ U(0, 1) is greater than the texture's alpha and ignoring
    /// the triangle if it is.
    fn cast_shadow_ray(&self, ray: Ray, light_distance: f32, accel_index: usize) -> bool {
        let mut distance = light_distance;
        let mut ray = ray;

        while let TraceResult::Hit {
            t,
            u,
            v,
            triangle_index,
        } = self.trace(&ray, accel_index)
        {
            // Never shadowed
            if t > distance {
                break;
            }

            let triangle = &self.triangles[triangle_index as usize];
            let material = &self.materials[triangle.material_index as usize];

            if let Some(i) = material.base_color_texture {
                if self.textures[i].format == Format::Rgb {
                    return true;
                }
            } else {
                return true;
            }

            let verts = [
                &self.verts[triangle.index1 as usize],
                &self.verts[triangle.index2 as usize],
                &self.verts[triangle.index3 as usize],
            ];

            let has_texture_coords = verts[0].tex_coord.is_some();

            let w = 1.0 - u - v;

            let texture_coordinates = if has_texture_coords {
                w * verts[0].tex_coord.unwrap() + u * verts[1].tex_coord.unwrap() + v * verts[2].tex_coord.unwrap()
            } else {
                vec2(0.0, 0.0)
            };

            if thread_rng().gen::<f32>() > material.sample_alpha(texture_coordinates, &self.textures) {
                // Cast another ray from slightly further than where we hit
                let hit_pos = w * verts[0].position + u * verts[1].position.to_vec() + v * verts[2].position.to_vec();
                let offset = 0.0002;
                ray.origin = hit_pos + offset * ray.direction;
                distance -= t + offset;
            } else {
                return true;
            }
        }

        false
    }

    fn trace(&self, ray: &Ray, accel_index: usize) -> TraceResult {
        self.accel_structures[accel_index].intersect(ray, &self.verts, &self.triangles)
    }

    /// From Ray Tracing Gems chapter 6: "A Fast and Robust Method for Avoiding Self-Intersection"
    fn offset_hit_pos(pos: Point3<f32>, normal: Vector3<f32>) -> Point3<f32> {
        const ORIGIN: f32 = 1.0 / 32.0;
        const FLOAT_SCALE: f32 = 1.0 / 65536.0;
        const INT_SCALE: f32 = 256.0;

        let offset_int = vec3(
            (INT_SCALE * normal.x) as i32,
            (INT_SCALE * normal.y) as i32,
            (INT_SCALE * normal.z) as i32,
        );

        let scale = |p: f32, o: i32| {
            use crate::util::bit_hacks::{f32_as_i32, i32_as_f32};

            i32_as_f32(f32_as_i32(p) + p.signum() as i32 * o)
        };

        let p_i = vec3(
            scale(pos.x, offset_int.x),
            scale(pos.y, offset_int.y),
            scale(pos.z, offset_int.z),
        );

        let e = |pos: f32, normal, p_i| {
            if pos.abs() < ORIGIN {
                pos + FLOAT_SCALE * normal
            } else {
                p_i
            }
        };

        point3(
            e(pos.x, normal.x, p_i.x),
            e(pos.y, normal.y, p_i.y),
            e(pos.z, normal.z, p_i.z),
        )
    }
}
