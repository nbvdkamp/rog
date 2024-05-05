#[allow(dead_code)]
#[derive(Debug)]
pub enum Error {
    Eframe(eframe::Error),
    ImageLoad(renderer::raytracer::file_formatting::Error),
    Io(std::io::Error),
}
