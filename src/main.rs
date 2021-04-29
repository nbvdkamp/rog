use glfw::{Context as _, WindowEvent};

use luminance_glfw::GlfwSurface;
use luminance_windowing::{WindowDim, WindowOpt};
use luminance_derive::{Semantics, Vertex, UniformInterface};
use luminance_front::pipeline::PipelineState;
use luminance_front::render_state::RenderState;
use luminance_front::context::GraphicsContext;
use luminance_front::tess::{Mode, Tess, TessError, Interleaved};
use luminance::shader::Uniform;
use luminance_front::Backend;

use cgmath::{perspective, EuclideanSpace, Matrix4, Point3, Rad, Vector3, Vector4};

use std::process::exit;
use std::time::Instant;
use std::path::Path;

fn main() {
    let dim = WindowDim::Windowed {
        width: 960,
        height: 540,
    };

    let surface = GlfwSurface::new_gl33("Window Title", WindowOpt::default().set_dim(dim));

    match surface {
        Ok(surface) => {
            eprintln!("Graphics surface created");
            main_loop(surface);
        }
        Err(e) => {
            eprintln!("Could not create graphics surface:\n{}", e);
            exit(1);
        }
    }
}

#[derive(Copy, Clone, Debug, Semantics)]
pub enum VertexSemantics {
    #[sem(name = "position", repr = "[f32; 3]", wrapper = "VertexPosition")]
    Position,   
}

#[derive(Copy, Clone, Vertex)]
#[vertex(sem = "VertexSemantics")]
pub struct Vertex {
    #[allow(dead_code)]
    position: VertexPosition,
}

type VertexIndex = u32;
pub struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<VertexIndex>,
}

impl Mesh {
    fn to_tess<C>(&self, context: &mut C) -> Result<Tess<Vertex, VertexIndex, (), Interleaved>, TessError>
    where
        C: GraphicsContext<Backend = Backend>,
    {
        context
            .new_tess()
            .set_mode(Mode::Triangle)
            .set_vertices(self.vertices.clone())
            .set_indices(self.indices.clone())
            .build()
    }
}

pub struct Scene {
    meshes: Vec<Mesh>,
}

impl Scene {
    fn load<P>(path: P) -> Result<Self, String>
    where
        P: AsRef<Path>,
    {
        if let Ok((document, buffers, _)) = gltf::import(path) {
            let mut meshes = Vec::<Mesh>::new();
            
            for scene in document.scenes() {
                for node in scene.nodes() {
                    if let Some(mesh) = node.mesh() {
                        for primitive in mesh.primitives() {
                            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                            let positions = {
                                let iter = reader
                                    .read_positions()
                                    .unwrap_or_else(||
                                        panic!("Primitive does not have POSITION attribute (mesh: {}, primitive: {})", mesh.index(), primitive.index())
                                    );
                                iter.collect::<Vec<_>>()
                            };

                            let vertices: Vec<Vertex> = positions
                                .into_iter()
                                .map(|position| {
                                    Vertex {
                                        position: position.into()
                                    }
                                }).collect();
                            
                            let indices: Vec<VertexIndex> = reader
                                .read_indices()
                                .map(|read_indices| {
                                    read_indices.into_u32().collect::<Vec<_>>()
                                })
                                .unwrap_or_else(||
                                    panic!("Primitive has no indices (mesh: {}, primitive: {})", mesh.index(), primitive.index())
                                );
                            
                            //TODO get normals and stuff.
                            meshes.push(Mesh { vertices, indices });
                        }
                    }
                }
            }

            Ok(Scene { meshes })
        }
        else {
            Err("Couldn't open glTF file.".into())
        }

    }
}

#[derive(Debug, UniformInterface)]
struct ShaderInterface {
    #[uniform(unbound)]
    u_projection: Uniform<[[f32; 4]; 4]>,
    #[uniform(unbound)]
    u_view: Uniform<[[f32; 4]; 4]>,
}


const VS_STR: &str = include_str!("passthrough.vs");
const FS_STR: &str = include_str!("color.fs");

const FOVY: Rad<f32> = Rad(std::f32::consts::FRAC_PI_2);
const Z_NEAR: f32 = 0.1;
const Z_FAR: f32 = 10.0;

fn main_loop(surface: GlfwSurface) {
    let mut context = surface.context;
    let events = surface.events_rx;
    let back_buffer = context.back_buffer().expect("back buffer");

    let start_t = Instant::now();

    let [width, height] = back_buffer.size();
    let projection = perspective(FOVY, width as f32 / height as f32, Z_NEAR, Z_FAR);
    let view = Matrix4::<f32>::look_at_rh(Point3::new(2., 2., 2.), Point3::origin(), Vector3::unit_y());

    let scene = Scene::load(Path::new("res/cube.glb")).unwrap();
    
    let mut program = context
        .new_shader_program::<VertexSemantics, (), ShaderInterface>()
        .from_strings(VS_STR, None, None, FS_STR)
        .unwrap()
        .ignore_warnings();

    'app: loop {
        context.window.glfw.poll_events();

        for (_, event) in glfw::flush_messages(&events) {
            match event {
                WindowEvent::Close => break 'app,
                _ => ()
            }
        }

        let m = &scene.meshes;
        let me = m.get(0).unwrap();
        let mesh = me.to_tess(&mut context).unwrap();

        let t = start_t.elapsed().as_millis() as f32 * 1e-3;
        let color = Vector4::new(t.cos(), t.sin(), 0.5, 1.);

        let render = context
            .new_pipeline_gate()
            .pipeline(
                &back_buffer,
                &PipelineState::default().set_clear_color(color.into()),
                |_, mut shd_gate| {
                    shd_gate.shade(&mut program, |mut iface, unif, mut rdr_gate| {
                        iface.set(&unif.u_projection, projection.into());
                        iface.set(&unif.u_view, view.into());

                        rdr_gate.render(&RenderState::default(), |mut tess_gate| {
                            tess_gate.render(&mesh)
                        })
                    })
                },
            )
            .assume();
        
        if render.is_ok() {
            context.window.swap_buffers();
        } else {
            break 'app;
        }
    }
}