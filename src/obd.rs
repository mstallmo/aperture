use crate::img;
pub use hashbrown::hash_map::HashMap;
use ndarray::prelude::*;
use std::error::Error;
use tensorflow::{
    Graph, ImportGraphDefOptions, Operation, Session, SessionOptions, SessionRunArgs, Tensor,
};

///Contains the Single Shot MultiBox Detector graph, the tensorflow session, and the image to inference.
///
/// ObjectDetection handles the initialization of the tensorflow session with the frozen graph. Converts the input image
/// from raw pixels to the tensor shape we need for input. Feeds the input into the input tensor of the graph
/// and returns the resulting output tensors in a HashMap.
pub struct ObjectDetection {
    graph: Graph,
    sess: Session,
    image: Box<img::DetectionImage>,
}

impl ObjectDetection {
    ///Initialize the tensorflow session with the frozen graph definition.
    pub fn init() -> Self {
        let mut graph = Graph::new();
        let proto =
            include_bytes!("../models/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb");

        graph
            .import_graph_def(proto, &ImportGraphDefOptions::new())
            .unwrap();
        let sess = Session::new(&SessionOptions::new(), &graph).unwrap();

        ObjectDetection {
            graph,
            sess,
            image: Box::new(img::GenericImage::default()),
        }
    }
    ///Pass in the input image to be inferenced on. Input must implement the DetectionImage trait
    pub fn input<I: img::DetectionImage + 'static>(&mut self, image: I) {
        self.image = Box::new(image);
    }

    fn input_transform(&self) -> Result<(Operation, Tensor<u8>), Box<Error>> {
        let image_dimension = self.image.dimension();
        let image_array = Array::from_shape_vec(
            (
                image_dimension.height as usize,
                image_dimension.width as usize,
                3,
            ),
            self.image.pixel_buffer().to_vec(),
        )?;
        let image_array_expanded = image_array.insert_axis(Axis(0));

        let image_tensor_op = self.graph.operation_by_name_required("image_tensor")?;
        let input_image_tensor = Tensor::new(&[
            1,
            u64::from(image_dimension.height),
            u64::from(image_dimension.width),
            3,
        ])
        .with_values(image_array_expanded.as_slice().unwrap())?;

        Ok((image_tensor_op, input_image_tensor))
    }
    ///Run the inference on the inputted image transforming the image to the shape of the image input tensor,
    ///performing the inferencing, and mapping the output tensors into the returned HashMap.
    pub fn run(&mut self) -> Result<HashMap<&str, Tensor<f32>>, Box<Error>> {
        let (image_tensor_op, input_image_tensor) = self.input_transform()?;

        let mut session_args = SessionRunArgs::new();
        session_args.add_feed(&image_tensor_op, 0, &input_image_tensor);

        let num_detections = self.graph.operation_by_name_required("num_detections")?;
        let num_detections_token = session_args.request_fetch(&num_detections, 0);

        let classes = self.graph.operation_by_name_required("detection_classes")?;
        let classes_token = session_args.request_fetch(&classes, 0);

        let boxes = self.graph.operation_by_name_required("detection_boxes")?;
        let boxes_token = session_args.request_fetch(&boxes, 0);

        let scores = self.graph.operation_by_name_required("detection_scores")?;
        let scores_token = session_args.request_fetch(&scores, 0);

        self.sess.run(&mut session_args)?;

        let num_detections_tensor = session_args.fetch::<f32>(num_detections_token)?;
        let classes_tensor = session_args.fetch::<f32>(classes_token)?;
        let boxes_tensor = session_args.fetch::<f32>(boxes_token)?;
        let scores_tensor = session_args.fetch::<f32>(scores_token)?;

        let mut tensor_map = HashMap::new();
        tensor_map.insert("num_detections", num_detections_tensor);
        tensor_map.insert("detection_classes", classes_tensor);
        tensor_map.insert("detection_boxes", boxes_tensor);
        tensor_map.insert("detection_scores", scores_tensor);

        Ok(tensor_map)
    }
}
