# espial
A Rust crate for doing object detection using Tensorflow!

espial uses the ssd_mobilenet developed by Google for object detection. ssd_mobilenet provides a quick execution time as well as a small size to allow for the most diverse range of deployment of the crate.


## Usage
**Input:** Input images can be provided through img::GenericImage that can be found in this crate or by passing in the Image struct found in the [pison image crate](https://github.com/PistonDevelopers/image). To use other image types not listed here there is a public trait `DetectionImage` that is also accepted as input to the model. 


**Output:** Arrays containing the detection infomration found in the image. This information includes the bounding boxes of found objects, the confidence score for those found objects, the detection classes for the objects, and the number of detections in the image.


## Changelog


## Code of Conduct

## Contact
Mason Stallmo <masonstallmo@gmail.com>



