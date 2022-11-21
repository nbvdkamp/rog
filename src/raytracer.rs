use rand::{thread_rng, Rng};
use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use cgmath::{point2, point3, vec2, vec3, EuclideanSpace, InnerSpace, Point3, Vector2, Vector3, Vector4};
use crossbeam_deque::{Injector, Steal};

mod aabb;
pub mod acceleration;
mod axis;
mod bsdf;
mod file_formatting;
pub mod geometry;
mod ray;
mod sampling;
mod scene_statistics;
mod shadingframe;
mod triangle;
pub mod working_image;

use crate::{
    camera::PerspectiveCamera,
    cie_data as CIE,
    environment::Environment,
    light::Light,
    material::Material,
    mesh::Vertex,
    raytracer::{file_formatting::Error, working_image::Pixel},
    render_settings::{ImageSettings, RenderSettings, TerminationCondition},
    scene::Scene,
    scene_version::SceneVersion,
    spectrum::Spectrumf32,
    texture::{CoefficientTexture, Texture},
};

use aabb::BoundingBox;
use acceleration::{structure::TraceResult, Accel, AccelerationStructures};
use bsdf::{mis2, Evaluation, Sample};
use geometry::{ensure_valid_reflection, orthogonal_vector};
use ray::Ray;
use sampling::{sample_item_from_cumulative_probabilities, tent_sample};
use scene_statistics::SceneStatistics;
use shadingframe::ShadingFrame;
use triangle::Triangle;
use working_image::WorkingImage;

pub struct Textures {
    pub base_color_coefficients: Vec<CoefficientTexture>,
    pub metallic_roughness: Vec<Texture>,
    pub transimission: Vec<Texture>,
    pub emissive: Vec<Texture>,
    pub normal: Vec<Texture>,
}

pub struct Raytracer {
    verts: Vec<Vertex>,
    triangles: Vec<Triangle>,
    materials: Vec<Material>,
    lights: Vec<Light>,
    textures: Textures,
    environment: Environment,
    stats: Option<SceneStatistics>,
    pub camera: PerspectiveCamera,
    pub accel_structures: AccelerationStructures,
}

pub struct RenderProgressReporting {
    pub report_interval: Duration,
    /// Completion, time elapsed
    pub report: Box<dyn Fn(f32, Duration)>,
}

pub struct ImageUpdateReporting {
    pub update_interval: Duration,
    /// Image, samples completed
    pub update: Box<dyn Fn(&WorkingImage, usize)>,
}

enum RadianceResult {
    Spectrum(Spectrumf32),
    SingleValue { value: f32, wavelength: f32 },
}
pub enum Wavelength {
    Undecided,
    Sampled { value: f32 },
}

impl Raytracer {
    pub fn new(
        scene: &Scene,
        textures: Textures,
        accel_structures_to_construct: &[Accel],
        use_visibility: bool,
        scene_version: Option<SceneVersion>,
    ) -> Self {
        let mut verts = Vec::new();
        let mut triangles = Vec::new();
        let mut triangle_bounds = Vec::new();
        let mut materials = Vec::new();

        // TODO: Fix borrowing to prevent having to clone everything
        for mesh in &scene.meshes {
            let start_index = verts.len() as u32;
            let material_index = materials.len() as u32;

            verts.append(&mut mesh.vertices.clone());
            materials.push(mesh.material.clone());

            for i in (0..mesh.indices.len()).step_by(3) {
                let index1 = mesh.indices[i] + start_index;
                let index2 = mesh.indices[i + 1] + start_index;
                let index3 = mesh.indices[i + 2] + start_index;

                let mut bounds = BoundingBox::new();
                bounds.add(verts[index1 as usize].position);
                bounds.add(verts[index2 as usize].position);
                bounds.add(verts[index3 as usize].position);

                triangles.push(Triangle {
                    index1,
                    index2,
                    index3,
                    material_index,
                });

                triangle_bounds.push(bounds);
            }
        }

        let scene_bounds = acceleration::helpers::compute_bounding_box(&verts);

        let mut result = Raytracer {
            verts,
            triangles,
            materials,
            lights: scene.lights.clone(),
            textures,
            environment: scene.environment.clone(),
            stats: None,
            camera: scene.camera,
            accel_structures: AccelerationStructures::default(),
        };

        let start = Instant::now();

        for accel in accel_structures_to_construct {
            result
                .accel_structures
                .construct(*accel, &result.verts, &result.triangles, &triangle_bounds)
                .unwrap();
        }

        let l = accel_structures_to_construct.len();
        println!(
            "Constructed {} acceleration structure{} in {} seconds",
            l,
            if l == 1 { "" } else { "s" },
            start.elapsed().as_secs_f32()
        );

        if use_visibility {
            let scene_version = scene_version.expect("scene_version should be passed when using visibility data");

            let dir = PathBuf::from("output/cache/");
            let mut path = dir.clone();

            path.push(
                scene_version
                    .filepath
                    .file_name()
                    .expect("scene file should have a name"),
            );
            path.set_extension("vis");

            let cached_stats = SceneStatistics::read_from_file(path.clone(), &scene_version);

            let stats = if let Ok(stats) = cached_stats {
                stats
            } else {
                // If we can't find matching cached visibilty data, compute it
                let mut stats = SceneStatistics::new(scene_bounds, scene_version);

                let start = Instant::now();
                stats.sample_visibility(&result, accel_structures_to_construct[0]);
                println!("Computed visibility map in {} seconds", start.elapsed().as_secs_f32());

                let start = Instant::now();
                stats.sample_materials(&result, &triangle_bounds);
                println!(
                    "Computed material averages in {} seconds",
                    start.elapsed().as_secs_f32()
                );

                stats.compute_visibility_weighted_material_sums();

                if let Err(e) = std::fs::create_dir_all(dir)
                    .map_err(|e| Error::IO(e))
                    .and(stats.write_to_file(path))
                {
                    println!("Writing visibility data to file failed: {e}");
                }

                stats
            };

            result.stats = Some(stats);
        }

        result
    }

    pub fn dump_visibility_data(&self) {
        if let Some(stats) = self.stats.as_ref() {
            stats.dump_visibility_image("output/vis.png");

            if let Err(e) = stats.dump_materials_as_rgb("output/materials.grid") {
                eprintln!("Couldn't write materials file: {e}")
            }
        }
    }

    pub fn render(
        &self,
        settings: &RenderSettings,
        reporting: Option<RenderProgressReporting>,
        image_reporting: Option<ImageUpdateReporting>,
        image: WorkingImage,
    ) -> (WorkingImage, f32) {
        let image_size = image.settings.size();
        let aspect_ratio = image_size.x as f32 / image_size.y as f32;
        let fov_factor = (self.camera.y_fov / 2.).tan();

        let start = Instant::now();
        let mut last_progress_report = start;
        let mut last_image_update = start;

        let cam_model = self.camera.model;
        let cam_pos4 = cam_model * Vector4::new(0., 0., 0., 1.);
        let camera_pos = Point3::from_homogeneous(cam_pos4);
        let image_settings = &image.settings.clone();
        let image = Arc::new(Mutex::new(image));

        let tile_size = 100;
        let tiles: &Injector<Tile> = &Injector::new();

        add_tiles_to_queue(tiles, image_size, tile_size, 0);
        let tiles_per_sample = tiles.len();

        match settings.termination_condition {
            TerminationCondition::SampleCount(samples) => {
                for sample in 1..samples {
                    add_tiles_to_queue(tiles, image_size, tile_size, sample);
                }
            }
            TerminationCondition::Time(_) => {
                // Always have one extra set of tiles in the queue to prevent threads being idle
                add_tiles_to_queue(tiles, image_size, tile_size, 1);
            }
        }

        let finished = Arc::new(Mutex::new(false));
        let current_sample = Arc::new(Mutex::new(0usize));
        let tiles_completed_for_current_sample = Arc::new(Mutex::new(0usize));

        thread::scope(|s| {
            for i in 0..settings.thread_count {
                let image = Arc::clone(&image);
                let finished = Arc::clone(&finished);
                let current_sample = Arc::clone(&current_sample);
                let tiles_completed_for_current_sample = Arc::clone(&tiles_completed_for_current_sample);

                let work = move || {
                    'work: loop {
                        let tile = loop {
                            match tiles.steal() {
                                Steal::Success(tile) => break tile,
                                Steal::Empty => {
                                    if *finished.lock().unwrap() {
                                        break 'work;
                                    }
                                }
                                Steal::Retry => {}
                            }
                        };

                        let mut buffer = vec![
                            Pixel {
                                samples: 0,
                                spectrum: Spectrumf32::constant(0.0)
                            };
                            tile_size * tile_size
                        ];

                        for y in tile.start.y..tile.end.y {
                            for x in tile.start.x..tile.end.x {
                                let i = x - tile.start.x;
                                let j = y - tile.start.y;
                                let pixel = &mut buffer[tile_size * j + i];

                                let mut offset = vec2(0.5, 0.5);

                                if tile.sample > 0 || settings.intermediate_read_path.is_some() {
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

                                match self.radiance(ray, settings, &image_settings) {
                                    RadianceResult::Spectrum(spectrum) => {
                                        pixel.spectrum += spectrum;
                                        pixel.samples += Spectrumf32::RESOLUTION as u32;
                                    }
                                    RadianceResult::SingleValue { value, wavelength } => {
                                        // wavelength is sampled in range so no bounds checks necessary
                                        pixel.spectrum.add_at_wavelength_lerp(value, wavelength);
                                        pixel.samples += 1;
                                    }
                                }
                            }
                        }

                        while *current_sample.lock().unwrap() < tile.sample {
                            let sleep_duration = Duration::from_millis(10);
                            thread::sleep(sleep_duration);

                            if *finished.lock().unwrap() {
                                break 'work;
                            }
                        }

                        let mut image = image.lock().unwrap();

                        for y in tile.start.y..tile.end.y {
                            for x in tile.start.x..tile.end.x {
                                let i = x - tile.start.x;
                                let j = y - tile.start.y;
                                let pixel = &buffer[tile_size * j + i];
                                image.pixels[image_size.x * y + x].spectrum += pixel.spectrum;
                                image.pixels[image_size.x * y + x].samples += pixel.samples;
                            }
                        }

                        *tiles_completed_for_current_sample.lock().unwrap() += 1;
                    }
                };

                if let Err(e) = thread::Builder::new()
                    .name(format!("Render thread {i}"))
                    .spawn_scoped(s, work)
                {
                    eprintln!("Unable to spawn thread {i}: {e}");
                    std::process::exit(-1);
                };
            }

            let terminate = || match settings.termination_condition {
                TerminationCondition::SampleCount(_) => tiles.is_empty(),
                TerminationCondition::Time(time_limit) => start.elapsed() >= time_limit,
            };

            while !terminate() {
                let sleep_duration = Duration::from_millis(1);
                thread::sleep(sleep_duration);

                let mut completed = tiles_completed_for_current_sample.lock().unwrap();

                let mut current_sample = current_sample.lock().unwrap();
                let current_sample_finished = *completed == tiles_per_sample;

                if let Some(reporting) = &reporting {
                    let elapsed = start.elapsed();

                    if last_progress_report.elapsed() > reporting.report_interval {
                        let completion = match settings.termination_condition {
                            TerminationCondition::SampleCount(samples) => {
                                let total_tiles = tiles_per_sample * samples;
                                let completed = total_tiles - tiles.len();
                                completed as f32 / total_tiles as f32
                            }
                            TerminationCondition::Time(time_limit) => elapsed.as_secs_f32() / time_limit.as_secs_f32(),
                        };
                        (reporting.report)(completion, elapsed);
                        last_progress_report = Instant::now();
                    }
                }

                if let Some(reporting) = &image_reporting {
                    if current_sample_finished && last_image_update.elapsed() > reporting.update_interval {
                        (reporting.update)(&image.lock().unwrap(), *current_sample);
                        last_image_update = Instant::now();
                    }
                }

                if current_sample_finished {
                    *completed = 0;
                    *current_sample += 1;

                    if let TerminationCondition::Time(_) = settings.termination_condition {
                        add_tiles_to_queue(tiles, image_size, tile_size, *current_sample + 1);
                    }
                }
            }

            *finished.lock().unwrap() = true;
        });

        // Errors when the lock has multiple owners but the scope should guarantee that never happens
        let lock = Arc::try_unwrap(image).ok().unwrap();
        let image = lock.into_inner().expect("Cannot unlock image mutex");

        (image, start.elapsed().as_secs_f32())
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

    fn radiance(&self, ray: Ray, settings: &RenderSettings, image_settings: &ImageSettings) -> RadianceResult {
        let mut result = Spectrumf32::constant(0.0);
        let mut path_weight = Spectrumf32::constant(1.0);
        let mut ray = ray;

        let mut wavelength = Wavelength::Undecided;

        let num_lights = self.lights.len();
        let mut depth = 0;

        // Hard cap bounces to prevent endless bouncing inside perfectly reflective surfaces
        while !image_settings.max_depth_reached(depth) && depth < 100 {
            let TraceResult::Hit {
                triangle_index, u, v, ..
            } = self.trace(&ray, settings.accel_structure) else {
                result += path_weight * self.environment.sample(ray.direction);
                break;
            };

            let triangle = &self.triangles[triangle_index as usize];
            let material = &self.materials[triangle.material_index as usize];

            let verts = [
                &self.verts[triangle.index1 as usize],
                &self.verts[triangle.index2 as usize],
                &self.verts[triangle.index3 as usize],
            ];

            let w = 1.0 - u - v;
            let hit_pos = w * verts[0].position + u * verts[1].position.to_vec() + v * verts[2].position.to_vec();

            if let Wavelength::Undecided = wavelength {
                if image_settings.always_sample_single_wavelength {
                    let value = if let Some(stats) = &self.stats {
                        let voxel_index = stats.get_voxel_index(hit_pos);

                        let distribution = stats.spectral_distributions[voxel_index]
                            .as_ref()
                            .expect("voxel should have distributions");

                        let i = sample_item_from_cumulative_probabilities(&distribution.cumulative_probabilities.data)
                            .expect("data can't be empty");

                        path_weight /= distribution.probabilities.data[i] * Spectrumf32::RESOLUTION as f32;

                        const STEP_SIZE: f32 = CIE::LAMBDA_RANGE / Spectrumf32::RESOLUTION as f32;

                        CIE::LAMBDA_MIN + STEP_SIZE * (i as f32 + thread_rng().gen::<f32>())
                    } else {
                        // Sample uniformly
                        CIE::LAMBDA_MIN + CIE::LAMBDA_RANGE * thread_rng().gen::<f32>()
                    };

                    wavelength = Wavelength::Sampled { value };
                }
            }

            let has_texture_coords = verts[0].tex_coord.is_some();

            let texture_coordinates = if has_texture_coords {
                w * verts[0].tex_coord.unwrap()
                    + u * verts[1].tex_coord.unwrap().to_vec()
                    + v * verts[2].tex_coord.unwrap().to_vec()
            } else {
                point2(0.0, 0.0)
            };

            let mut mat_sample = material.sample(texture_coordinates, &self.textures);

            if thread_rng().gen::<f32>() > mat_sample.alpha {
                // Offset to the back of the triangle
                ray.origin = hit_pos + ray.direction * 0.0002;

                // Don't count alpha hits for max bounces
                continue;
            }

            if image_settings.enable_dispersion && mat_sample.transmission > 0.0 {
                let lambda;

                match wavelength {
                    Wavelength::Undecided => {
                        lambda = CIE::LAMBDA_MIN + CIE::LAMBDA_RANGE * thread_rng().gen::<f32>();
                        wavelength = Wavelength::Sampled { value: lambda }
                    }
                    Wavelength::Sampled { value } => lambda = value,
                }

                mat_sample.ior = mat_sample.cauchy_coefficients.ior_for_wavelength(lambda);
            }

            let edge1 = verts[0].position - verts[1].position;
            let edge2 = verts[0].position - verts[2].position;
            let mut geom_normal = edge1.cross(edge2).normalize();

            // Interpolate the vertex normals
            let mut normal = (w * verts[0].normal + u * verts[1].normal + v * verts[2].normal).normalize();
            let mut tangent = (w * verts[0].tangent + u * verts[1].tangent + v * verts[2].tangent).normalize();

            // Handle the rare case where a bad tangent (linearly dependent with the normal) causes NaNs
            if normal == tangent || -normal == tangent {
                tangent = orthogonal_vector(normal);
            }

            // Flip the computed geometric normal to the same side as the interpolated vertex normal.
            if geom_normal.dot(normal) < 0.0 {
                geom_normal = -geom_normal;
            }

            let backfacing = geom_normal.dot(-ray.direction) < 0.0;

            if backfacing {
                normal = -normal;

                // Swap IORs
                (mat_sample.ior, mat_sample.medium_ior) = (mat_sample.medium_ior, mat_sample.ior);
            }

            let mut frame = ShadingFrame::new_with_tangent(normal, tangent);

            let shading_normal = mat_sample
                .shading_normal
                .map(|n| ensure_valid_reflection(geom_normal, -ray.direction, frame.to_global(n)));

            if let Some(shading_normal) = shading_normal {
                frame = ShadingFrame::new(shading_normal);
            }

            let local_outgoing = frame.to_local(-ray.direction);

            let sample = bsdf::sample(&mat_sample, local_outgoing);

            // Next event estimation (directly sampling lights)
            if num_lights > 0 {
                let light = &self.lights[rand::thread_rng().gen_range(0..num_lights)];
                let light_sample = light.sample(hit_pos);

                let offset_direction = light_sample.direction.dot(geom_normal).signum() * geom_normal;
                let offset_hit_pos = Raytracer::offset_hit_pos(hit_pos, offset_direction);

                if light_sample.distance <= light.range {
                    let shadow_ray = Ray {
                        origin: offset_hit_pos,
                        direction: light_sample.direction,
                    };

                    let shadowed = self.is_ray_obstructed(shadow_ray, light_sample.distance, settings.accel_structure);

                    if !shadowed {
                        let local_incident = frame.to_local(light_sample.direction);
                        let eval = bsdf::eval(&mat_sample, local_outgoing, local_incident);

                        if let Evaluation::Evaluation { weight: bsdf, pdf } = eval {
                            let light_pick_prob = 1.0 / num_lights as f32;
                            let light_pdf = light_pick_prob * light_sample.pdf;

                            let mis_weight = if light_sample.use_mis {
                                mis2(light_pdf, pdf)
                            } else {
                                1.0
                            };

                            let shadow_terminator = shading_normal.map_or(1.0, |shading_normal| {
                                bump_shading_factor(normal, shading_normal, light_sample.direction)
                            });

                            result += path_weight
                                    // * mat_sample.base_color_spectrum
                                    * mis_weight
                                    * light_sample.intensity
                                    * shadow_terminator
                                    * bsdf
                                / light_pdf
                                * light.spectrum;
                        }
                    }
                }
            }

            let (local_bounce_dir, bsdf, pdf) = match sample {
                Sample::Sample { incident, weight, pdf } => (incident, weight, pdf),
                Sample::Null => break,
            };

            let offset_direction = if backfacing { -1.0 } else { 1.0 } * local_bounce_dir.z.signum() * geom_normal;
            let offset_hit_pos = Raytracer::offset_hit_pos(hit_pos, offset_direction);

            let bounce_dir = frame.to_global(local_bounce_dir);

            let shadow_terminator = shading_normal.map_or(1.0, |shading_normal| {
                bump_shading_factor(normal, shading_normal, bounce_dir)
            });

            path_weight *= bsdf * shadow_terminator / pdf;

            let continue_prob = path_weight.max_value().max(1.0);

            if thread_rng().gen::<f32>() < continue_prob {
                path_weight /= continue_prob;
            } else {
                break;
            }

            ray = Ray {
                origin: offset_hit_pos,
                direction: bounce_dir,
            };

            depth += 1;
        }

        match wavelength {
            Wavelength::Undecided => RadianceResult::Spectrum(result),
            Wavelength::Sampled { value: wavelength } => RadianceResult::SingleValue {
                value: result.at_wavelength_lerp(wavelength),
                wavelength,
            },
        }
    }

    /// Checks if distance to the nearest obstructing triangle is less than the distance
    /// Handles alpha by checking if R ~ U(0, 1) is greater than the texture's alpha and ignoring
    /// the triangle if it is.
    fn is_ray_obstructed(&self, ray: Ray, distance: f32, accel: Accel) -> bool {
        let mut distance = distance;
        let mut ray = ray;

        while let TraceResult::Hit {
            t,
            u,
            v,
            triangle_index,
        } = self.trace(&ray, accel)
        {
            // Never shadowed
            if t > distance {
                break;
            }

            let triangle = &self.triangles[triangle_index as usize];
            let material = &self.materials[triangle.material_index as usize];

            if let Some(tex_ref) = material.base_color_texture {
                if !self.textures.base_color_coefficients[tex_ref.index].has_alpha {
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
                w * verts[0].tex_coord.unwrap()
                    + u * verts[1].tex_coord.unwrap().to_vec()
                    + v * verts[2].tex_coord.unwrap().to_vec()
            } else {
                point2(0.0, 0.0)
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

    fn trace(&self, ray: &Ray, accel: Accel) -> TraceResult {
        self.accel_structures
            .get(accel)
            .intersect(ray, &self.verts, &self.triangles)
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

pub fn render_and_save<P>(raytracer: &Raytracer, render_settings: &RenderSettings, image: WorkingImage, path: P)
where
    P: AsRef<Path>,
{
    let report_progress = |completion, time_elapsed: Duration| {
        let completion_percentage = 100.0 * completion;
        let total_time = time_elapsed.as_secs_f32() / completion;
        let time_remaining = total_time * (1.0 - completion);
        print!("\r\x1b[2K Approximately {completion_percentage:.2}% complete, {time_remaining:.2} seconds remaining");
        std::io::stdout().flush().unwrap();
    };

    let progress = Some(RenderProgressReporting {
        report_interval: Duration::from_secs(1),
        report: Box::new(report_progress),
    });

    let (image, time_elapsed) = raytracer.render(&render_settings, progress, None, image);
    println!("\r\x1b[2KFinished rendering in {time_elapsed} seconds");

    image.save_as_rgb(path);

    if let Some(path) = &render_settings.intermediate_write_path {
        match image.write_to_file(path) {
            Ok(()) => println!("Saved intermediate file successfully"),
            Err(e) => {
                eprintln!("Failed to save intermediate file: {e}");
            }
        };
    }
}

/// From Chiang et al. 2019, 'Taming the Shadow Terminator'
fn bump_shading_factor(
    geometric_normal: Vector3<f32>,
    shading_normal: Vector3<f32>,
    light_direction: Vector3<f32>,
) -> f32 {
    let g_dot_i = geometric_normal.dot(light_direction);
    let s_dot_i = shading_normal.dot(light_direction);
    let g_dot_s = geometric_normal.dot(shading_normal);
    let g = (g_dot_i / (s_dot_i * g_dot_s)).clamp(0.0, 1.0);

    // Hermite interpolation
    -(g * g * g) + (g * g) + g
}

struct Tile {
    start: Vector2<usize>,
    end: Vector2<usize>,
    sample: usize,
}

fn add_tiles_to_queue(tiles: &Injector<Tile>, image_size: Vector2<usize>, tile_size: usize, sample: usize) {
    for x_start in (0..image_size.x).step_by(tile_size) {
        let x_end = (x_start + tile_size).min(image_size.x);

        for y_start in (0..image_size.y).step_by(tile_size) {
            let y_end = (y_start + tile_size).min(image_size.y);

            tiles.push(Tile {
                start: vec2(x_start, y_start),
                end: vec2(x_end, y_end),
                sample,
            });
        }
    }
}
