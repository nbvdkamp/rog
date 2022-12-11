use rayon::prelude::*;

use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use serde::{Deserialize, Serialize};

use crate::{
    color::RGBf32,
    raytracer::file_formatting::{Error, SectionHeader},
    render_settings::ImageSettings,
    scene_version::SceneVersion,
    spectrum::Spectrumf32,
    util::{align_to, save_image},
};

const FORMAT_VERSION: u32 = 1;
const TAG: &str = "SPECTRAL IMG";
const JSON_TAG: &str = "JSON";
const PIXELS_TAG: &str = "PIX ";
static_assertions::const_assert_eq!(TAG.len() % 4, 0);
static_assertions::const_assert_eq!(JSON_TAG.len() % 4, 0);
static_assertions::const_assert_eq!(PIXELS_TAG.len() % 4, 0);

#[derive(Serialize, Deserialize)]
pub struct WorkingImage {
    pub settings: ImageSettings,
    #[serde(skip)]
    pub pixels: Vec<Pixel>,
}

#[derive(Clone)]
pub struct Pixel {
    pub spectrum: Spectrumf32,
    pub samples: u32,
}

impl WorkingImage {
    pub fn new(settings: ImageSettings) -> Self {
        WorkingImage {
            pixels: vec![
                Pixel {
                    spectrum: Spectrumf32::constant(0.0),
                    samples: 0
                };
                settings.width * settings.height
            ],
            settings,
        }
    }

    pub fn save_as_rgb<P>(&self, path: P)
    where
        P: AsRef<Path>,
    {
        let buffer = self.to_rgb_buffer();
        save_image(&buffer, self.settings.size(), path)
    }

    pub fn to_rgb_buffer(&self) -> Vec<RGBf32> {
        self.pixels
            .par_iter()
            .map(|pixel| {
                if pixel.spectrum.data.iter().any(|v| v.is_nan()) {
                    return RGBf32::new(1.0, 0.5, 0.0);
                }
                let spectrum = pixel.spectrum * Spectrumf32::RESOLUTION as f32 / pixel.samples as f32;

                let c = spectrum.to_srgb().linear_to_srgb();
                if c.has_nan_component() {
                    RGBf32::new(0.0, 1.0, 0.0)
                } else {
                    c
                }
            })
            .collect()
    }

    pub fn write_to_file<P>(&self, path: P) -> Result<(), Error>
    where
        P: AsRef<Path>,
    {
        use Error::{Serde, IO};
        let mut file = File::create(path).map_err(IO)?;

        let json = serde_json::to_string(self).map_err(Serde)?;
        let bytes_to_pad = align_to(json.len(), 4) - json.len();
        let json = format!("{}{}", json, " ".repeat(bytes_to_pad));

        let pixels_size = self.pixels.len() * std::mem::size_of::<Pixel>();

        let size = FileHeader::SIZE
            + JSON_TAG.len()
            + PIXELS_TAG.len()
            + 2 * std::mem::size_of::<SectionHeader>()
            + json.len()
            + pixels_size;

        let header = FileHeader {
            format_version: FORMAT_VERSION,
            total_size: size as u64,
            spectrum_resolution: Spectrumf32::RESOLUTION as u32,
        };

        header.to_writer(&mut file).map_err(IO)?;

        let json_header = SectionHeader::new(json.len(), JSON_TAG);
        json_header.to_writer(&mut file).map_err(IO)?;

        file.write_all(json.as_bytes()).map_err(IO)?;

        let pixels_header = SectionHeader::new(pixels_size, PIXELS_TAG);
        pixels_header.to_writer(&mut file).map_err(IO)?;

        let pixels_buffer: &[u8] =
            unsafe { std::slice::from_raw_parts(self.pixels.as_ptr() as *const u8, pixels_size) };
        file.write_all(pixels_buffer).map_err(IO)?;

        Ok(())
    }

    pub fn read_from_file<P>(path: P, expected_scene_version: &Option<SceneVersion>) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        use Error::{Serde, IO};
        let mut file = File::open(path).map_err(IO)?;

        let _header = FileHeader::from_reader(&mut file)?;

        let json_header = SectionHeader::from_reader(&mut file, JSON_TAG, None)?;

        //Read json
        let mut json_buffer = vec![0; json_header.size as usize];
        file.read_exact(&mut json_buffer).map_err(IO)?;

        let mut image = serde_json::from_slice::<WorkingImage>(&json_buffer).map_err(Serde)?;

        if let Some(expected) = expected_scene_version {
            if image
                .settings
                .scene_version
                .as_ref()
                .map_or(true, |v| v.hash != expected.hash)
            {
                return Err(Error::SceneMismatch);
            }
        }

        let expected_pixel_count = image.settings.width * image.settings.height;
        let expected_pixels_size = expected_pixel_count * std::mem::size_of::<Pixel>();

        let _pixels_header = SectionHeader::from_reader(&mut file, PIXELS_TAG, Some(expected_pixels_size))?;

        image.pixels = vec![
            Pixel {
                spectrum: Spectrumf32::constant(0.0),
                samples: 0
            };
            expected_pixel_count
        ];

        unsafe {
            file.read_exact(std::slice::from_raw_parts_mut(
                image.pixels.as_ptr() as *mut u8,
                expected_pixels_size,
            ))
            .map_err(IO)?;
        }

        Ok(image)
    }
}

struct FileHeader {
    pub format_version: u32,
    pub total_size: u64,
    pub spectrum_resolution: u32,
}

impl FileHeader {
    const SIZE: usize = TAG.len() + std::mem::size_of::<Self>();

    pub fn to_writer<W>(self, writer: &mut W) -> Result<(), std::io::Error>
    where
        W: Write,
    {
        write!(writer, "{}", TAG)?;
        writer.write_u32::<LittleEndian>(self.format_version)?;
        writer.write_u64::<LittleEndian>(self.total_size)?;
        writer.write_u32::<LittleEndian>(self.spectrum_resolution)?;
        Ok(())
    }

    pub fn from_reader<R: Read>(reader: &mut R) -> Result<Self, Error> {
        use Error::IO;
        let mut tag = [0; TAG.len()];
        reader.read_exact(&mut tag).map_err(IO)?;

        if tag != TAG.as_bytes() {
            return Err(Error::TagMismatch {
                expected: TAG.as_bytes().to_vec(),
                actual: tag.to_vec(),
            });
        }

        let format_version = reader.read_u32::<LittleEndian>().map_err(IO)?;
        if format_version != FORMAT_VERSION {
            return Err(Error::FormatVersionMismatch {
                current: FORMAT_VERSION,
                file: format_version,
            });
        }

        let total_size = reader.read_u64::<LittleEndian>().map_err(IO)?;

        let spectrum_resolution = reader.read_u32::<LittleEndian>().map_err(IO)?;
        if spectrum_resolution as usize != Spectrumf32::RESOLUTION {
            return Err(Error::ResolutionMismatch {
                current: Spectrumf32::RESOLUTION,
                file: spectrum_resolution as usize,
                name: "spectrum".to_string(),
            });
        }

        Ok(Self {
            format_version,
            total_size,
            spectrum_resolution,
        })
    }
}
