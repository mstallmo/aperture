#![deny(missing_docs)]
//! # Espial
//! A library for performing object detection utilizing the Single Shot MultiBox
//! Detector model created by google.
//!
//! Input for espial takes a type implementing the DetectionImage trait and performs object detection on
//! that image. The list of objects that can be detected via the label map for the model can be found
//! [here](https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt)
//!
//! Output is returned as a hash map of the detection tensors containing the detection boxes, detection classes
//! detection scores, and the number of detections. Each element of the returned HashMap can be accessed by the
//! name of the tensor in all lower case separated by an underscore. (e.x. detection_scores)

///Image module representing the input image to be inferenced.
pub mod img;
///ObjectDetection module that handles the setup and inferencing of the image data.
pub mod obd;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
