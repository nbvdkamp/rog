use std::fs::File;

use serde::{Deserialize, Serialize};
use sha2::{digest::DynDigest, Digest, Sha256};

#[derive(Serialize, Deserialize, Clone)]
pub struct SceneVersion {
    pub filename: String,
    pub hash: [u8; 32],
}

impl SceneVersion {
    pub fn new(filename: String) -> Result<Self, std::io::Error> {
        let mut file = File::open(&filename)?;
        let mut hasher = Sha256::new();

        let _ = std::io::copy(&mut file, &mut hasher)?;

        assert_eq!(hasher.output_size(), 32);
        let mut hash = [0; 32];
        hash.copy_from_slice(&hasher.finalize());

        Ok(SceneVersion { filename, hash })
    }
}
