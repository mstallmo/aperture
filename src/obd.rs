use crate::utils;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use tensorflow::{Graph, ImportGraphDefOptions, Session, SessionOptions, SessionRunArgs};

const SSD_MODEL_PATH: &str = "./models/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb";

pub struct ObjectDetection {
    graph: Graph,
    sess: Session,
    pixels: Vec<u8>,
}

impl ObjectDetection {
    pub fn init() -> Self {
        let mut graph = Graph::new();
        let mut proto = Vec::new();

        if !Path::new(SSD_MODEL_PATH).exists() {
            utils::download_model("./models/ssd_mobilenet_v1_coco_2017_11_17.tar.gz").unwrap();
        }

        File::open(SSD_MODEL_PATH)
            .unwrap()
            .read_to_end(&mut proto)
            .unwrap();
        graph
            .import_graph_def(&proto, &ImportGraphDefOptions::new())
            .unwrap();
        let sess = Session::new(&SessionOptions::new(), &graph).unwrap();
        ObjectDetection {
            graph,
            sess,
            pixels: Vec::new(),
        }
    }

    pub fn input(&mut self, image_pixels: Vec<u8>) {
        self.pixels = image_pixels;
    }

    pub fn run(&mut self, session_args: &mut SessionRunArgs) -> Result<(), Box<Error>> {
        self.sess.run(session_args)?;

        let num_detections = self.graph.operation_by_name_required("num_detections")?;

        Ok(())
    }
}
