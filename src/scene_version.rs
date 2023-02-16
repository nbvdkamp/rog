use std::{fs::File, path::PathBuf};

use derivative::Derivative;
use serde::{Deserialize, Serialize};
use sha2::{digest::DynDigest, Digest, Sha256};

#[derive(Serialize, Deserialize, Clone, Derivative)]
#[derivative(Debug)]
pub struct SceneVersion {
    pub filepath: PathBuf,
    #[derivative(Debug = "ignore")]
    pub hash: [u8; 32],
}

impl SceneVersion {
    pub fn new(filepath: PathBuf) -> Result<Self, std::io::Error> {
        let mut file = File::open(&filepath)?;
        let mut hasher = Sha256::new();

        let _ = std::io::copy(&mut file, &mut hasher)?;

        assert_eq!(hasher.output_size(), 32);
        let mut hash = [0; 32];
        hash.copy_from_slice(&hasher.finalize());

        Ok(SceneVersion { filepath, hash })
    }
}
