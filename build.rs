use flate2::read::GzDecoder;
use reqwest;
use std::fs::File;
use std::io::copy;
use std::path::Path;
use tar::Archive;

fn main() {
    if !Path::new("./models/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb").exists() {
        download_model("./models/ssd_mobilenet_v1_coco_2017_11_17.tar.gz");
    }
}

fn download_model<P: AsRef<Path>>(destination: P) {
    println!("Downloading model...");

    let file_path = Path::new(destination.as_ref());
    let mut response = reqwest::get("http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz").unwrap();
    let mut dest = {
        std::fs::create_dir_all("./models").unwrap();
        File::create(file_path).unwrap()
    };
    copy(&mut response, &mut dest).unwrap();

    unzip_archive(file_path).unwrap();
    std::fs::remove_file(file_path).unwrap();

    println!("Download finished!");
}

fn unzip_archive<P: AsRef<Path>>(file_path: P) -> Result<(), std::io::Error> {
    println!("Unzipping file...");
    let tar = GzDecoder::new(File::open(file_path)?);
    let mut archive = Archive::new(tar);
    archive.unpack("./models")?;
    println!("Unzipping complete!");
    Ok(())
}
