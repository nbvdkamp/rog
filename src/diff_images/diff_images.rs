mod structural_similarity;
use renderer::raytracer::{
    file_formatting::Error as FileFormatError,
    single_channel_image::SingleChannelImage,
    working_image::WorkingImage,
};
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

    output_error("MSE", image.mean_square_error(&reference));
    output_error("rgb_MSE", image.rgb_mean_square_error(&reference));
    output_error("relMSE", image.relative_mean_square_error(&reference));

    let img_grayscale = image.to_grayscale();
    let ref_grayscale = reference.to_grayscale();
    let grayscale_diff = &img_grayscale - &ref_grayscale;
    output_error("grayscale_MSE", &grayscale_diff * &grayscale_diff);
    output_error("MSSIM", structural_similarity(img_grayscale, ref_grayscale));

    Ok(())
}

fn output_error(name: &str, error_image: SingleChannelImage) {
    let mean = error_image.mean();
    println!("{name} {mean}");
}
