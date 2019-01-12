use crate::img;
use crate::utils;
pub use hashbrown::hash_map::HashMap;
use ndarray::prelude::*;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use tensorflow::{
    Graph, ImportGraphDefOptions, Operation, Session, SessionOptions, SessionRunArgs, Tensor,
};

const SSD_MODEL_PATH: &str = "./models/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb";

pub struct ObjectDetection<T: img::DetectionImage> {
    graph: Graph,
    sess: Session,
    image: T,
}

impl<T> ObjectDetection<T>
where
    T: img::DetectionImage,
{
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
            image: img::GenericImage::default(),
        }
    }

    pub fn input (&mut self, image: T) {
        self.image = image;
    }

    fn input_transform(&self) -> Result<(Operation, Tensor<u8>), Box<Error>> {
        let (width, height) = self.image.dimension();
        let image_array = Array::from_shape_vec(
            (height as usize, width as usize, 3),
            self.image.pixel_buffer().to_vec(),
        )?;
        let image_array_expanded = image_array.insert_axis(Axis(0));

        let image_tensor_op = self.graph.operation_by_name_required("image_tensor")?;
        let input_image_tensor = Tensor::new(&[1, height as u64, width as u64, 3])
            .with_values(image_array_expanded.as_slice().unwrap())?;

        Ok((image_tensor_op, input_image_tensor))
    }

    pub fn run(&mut self) -> Result<HashMap<&str, Tensor<f32>>, Box<Error>> {
        let (image_tensor_op, input_image_tensor) = self.input_transform()?;

        let mut session_args = SessionRunArgs::new();
        session_args.add_feed(&image_tensor_op, 0, &input_image_tensor);

        self.sess.run(&mut session_args)?;

        let output = self.output_transform(&mut session_args)?;

        Ok(output)
    }

    //TODO: Move request_fetch operations before the sess.run call
    fn output_transform(
        &self,
        session_args: &mut SessionRunArgs,
    ) -> Result<HashMap<&str, Tensor<f32>>, Box<Error>> {
        let num_detections = self.graph.operation_by_name_required("num_detections")?;
        let num_detections_token = session_args.request_fetch(&num_detections, 0);
        let num_detections_tensor = session_args.fetch::<f32>(num_detections_token)?;

        let classes = self.graph.operation_by_name_required("detection_classes")?;
        let classes_token = session_args.request_fetch(&classes, 0);
        let classes_tensor = session_args.fetch::<f32>(classes_token)?;

        let boxes = self.graph.operation_by_name_required("detection_boxes")?;
        let boxes_token = session_args.request_fetch(&boxes, 0);
        let boxes_tensor = session_args.fetch::<f32>(boxes_token)?;

        let scores = self.graph.operation_by_name_required("detection_scores")?;
        let scores_token = session_args.request_fetch(&scores, 0);
        let scores_tensor = session_args.fetch::<f32>(scores_token)?;

        let mut tensor_map = HashMap::new();
        tensor_map.insert("num_detections", num_detections_tensor);
        tensor_map.insert("detection_classes", classes_tensor);
        tensor_map.insert("detection_boxes", boxes_tensor);
        tensor_map.insert("detection_scores", scores_tensor);

        Ok(tensor_map)
    }
}
