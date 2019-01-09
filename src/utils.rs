use reqwest;
use std::fs::File;
use std::path::Path;
use std::io::copy;
use std::error::Error;
use tar::Archive;
use flate2::read::GzDecoder;

pub fn download_model<P: AsRef<Path>>(destination: P) -> Result<(), Box<Error>> {
    println!("Downloading model!");

    let file_path = Path::new(destination.as_ref());
    let mut response = reqwest::get("http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz")?;
    let mut dest = {
        std::fs::create_dir_all("./models")?;
        File::create(file_path)?
    };
    copy(&mut response, &mut dest)?;

    unzip_archive(file_path)?;
    std::fs::remove_file(file_path)?;

    println!("Download finished!");
    Ok(())
}

fn unzip_archive<P: AsRef<Path>>(file_path: P) -> Result<(), std::io::Error> {
    let tar = GzDecoder::new(File::open(file_path).unwrap());
    let mut archive = Archive::new(tar);
    archive.unpack("./models")?;
    Ok(())
}