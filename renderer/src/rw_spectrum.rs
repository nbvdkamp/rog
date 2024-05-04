use renderer::spectrum::{ReadError, Spectrumf32};

#[derive(Debug)]
enum Error {
    ArgumentCount(usize),
    ReadError(ReadError),
}

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        return Err(Error::ArgumentCount(args.len()));
    }

    let spectrum = Spectrumf32::read_from_csv(&args[1]).map_err(Error::ReadError)?;
    println!("{:#?}", spectrum.data);
    println!("{:#?}", (50.0 * spectrum).to_srgb().normalized());

    Ok(())
}
