#[macro_use]
extern crate impl_ops;
mod structural_similarity;
use renderer::raytracer::{file_formatting::Error as FileFormatError, working_image::WorkingImage};

use structural_similarity::SingleChannelImage;

use crate::structural_similarity::structural_similarity;

#[derive(Debug)]
enum Error {
    File(FileFormatError),
    DimensionMismatch,
    ArgumentCount(usize),
}

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        return Err(Error::ArgumentCount(args.len()));
    }

    use Error::File;
    let first_img = WorkingImage::read_from_file(&args[1], &None).map_err(File)?;
    let second_img = WorkingImage::read_from_file(&args[2], &first_img.settings.scene_version).map_err(File)?;

    if first_img.settings.size != second_img.settings.size {
        return Err(Error::DimensionMismatch);
    }

    let width = first_img.settings.size.x;
    let height = first_img.settings.size.y;

    let image_x = SingleChannelImage {
        data: first_img.pixels.iter().map(|p| p.result_spectrum().mean()).collect(),
        width,
        height,
    };

    let image_y = SingleChannelImage {
        data: second_img.pixels.iter().map(|p| p.result_spectrum().mean()).collect(),
        width,
        height,
    };

    let mean_square_error = first_img.mean_square_error(&second_img);
    let root_mean_square_error = mean_square_error.sqrt();
    let mean_structural_similarity = structural_similarity(image_x, image_y).mean();

    println!("MSE {mean_square_error}");
    println!("RMSE {root_mean_square_error}");
    println!("MSSIM {mean_structural_similarity}");

    Ok(())
}
