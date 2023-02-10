mod structural_similarity;
use renderer::raytracer::{file_formatting::Error as FileFormatError, working_image::WorkingImage};
use structural_similarity::structural_similarity;

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
    let image = WorkingImage::read_from_file(&args[1], &None).map_err(File)?;
    let reference = WorkingImage::read_from_file(&args[2], &image.settings.scene_version).map_err(File)?;

    if image.settings.size != reference.settings.size {
        return Err(Error::DimensionMismatch);
    }

    let mean_square_error = image.mean_square_error(&reference).mean();
    let relative_mean_square_error = image.relative_mean_square_error(&reference).mean();
    let mean_structural_similarity = structural_similarity(image.to_grayscale(), reference.to_grayscale()).mean();

    println!("MSE {mean_square_error}");
    println!("relMSE {relative_mean_square_error}");
    println!("MSSIM {mean_structural_similarity}");

    Ok(())
}
