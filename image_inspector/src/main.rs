use itertools::Itertools;

mod app;
mod error;

use error::Error;
use renderer::raytracer::working_image::WorkingImage;

fn main() -> Result<(), Error> {
    let args = std::env::args().collect_vec();

    let image = if args.len() > 1 {
        Some(WorkingImage::read_from_file(&args[1], &None).map_err(Error::ImageLoad)?)
    } else {
        None
    };

    app::run(image).map_err(Error::Eframe)?;

    Ok(())
}
