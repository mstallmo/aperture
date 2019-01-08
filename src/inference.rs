use crate::utils;
use std::fs::File;
use std::io::Read;
use tensorflow::{Graph, ImportGraphDefOptions, Session, SessionOptions, SessionRunArgs};

const SSD_MODEL_PATH: &str = "./models/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb";

pub struct Inference {
    graph: Graph,
    sess: Session,
}

impl Inference {
    pub fn init() -> Self {
        let mut graph = Graph::new();
        let mut proto = Vec::new();
        //TODO: check if file exists, if not download model file.
        //utils::download_model();
        File::open(SSD_MODEL_PATH)
            .unwrap()
            .read_to_end(&mut proto)
            .unwrap();
        graph
            .import_graph_def(&proto, &ImportGraphDefOptions::new())
            .unwrap();
        let sess = Session::new(&SessionOptions::new(), &graph).unwrap();
        Inference { graph, sess }
    }

    pub fn run(&mut self, session_args: &mut SessionRunArgs) -> Result<(), Err>{
        self.sess.run(session_args)?;

        let num_detections = self.graph.operation_by_name_required("num_detections")?;
    }
}
