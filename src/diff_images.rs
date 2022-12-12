use renderer::raytracer::{file_formatting::Error as FileFormatError, working_image::WorkingImage};

#[derive(Debug)]
enum Error {
    File(FileFormatError),
    DiffModeParsing(String),
    ArgumentCount(usize),
}

enum DiffMode {
    MeanSquareError,
    RootMeanSquareError,
}

impl DiffMode {
    fn from_str(input: &str) -> Result<Self, Error> {
        match input {
            "MSE" => Ok(DiffMode::MeanSquareError),
            "RMSE" => Ok(DiffMode::RootMeanSquareError),
            _ => Err(Error::DiffModeParsing(input.to_string())),
        }
    }
}

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 4 {
        return Err(Error::ArgumentCount(args.len()));
    }

    let mode = DiffMode::from_str(&args[1])?;

    use Error::File;
    let first_img = WorkingImage::read_from_file(&args[2], &None).map_err(File)?;
    let second_img = WorkingImage::read_from_file(&args[3], &first_img.settings.scene_version).map_err(File)?;

    let error = match mode {
        DiffMode::MeanSquareError => first_img.mean_square_error(&second_img),
        DiffMode::RootMeanSquareError => first_img.mean_square_error(&second_img).sqrt(),
    };

    println!("{error}");

    Ok(())
}
