use image::GenericImageView;

///Represents generic image data. `width` and `height` are the width and height of the image and
///`pixel_buffer` is the raw pixel information of the image.
#[derive(Default, Clone)]
pub struct GenericImage {
    width: u32,
    height: u32,
    pixel_buffer: Vec<u8>,
}

impl GenericImage {
    ///creates a new GenericImage with the given width, height, and pixel data
    pub fn new(width: u32, height: u32, pixel_buffer: Vec<u8>) -> Self {
        GenericImage {
            width,
            height,
            pixel_buffer,
        }
    }
}

///Image data to be passed to the input tensor. Provides handles to the dimensions of the image and
/// the raw pixels of the image.
pub trait DetectionImage {
    ///Returns the dimensions of the image in the (width, height) pattern.
    fn dimension(&self) -> (u32, u32);
    ///Returns a vector containing the raw pixel information.
    fn pixel_buffer(&self) -> Vec<u8>;
}

impl DetectionImage for GenericImage {
    fn dimension(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn pixel_buffer(&self) -> Vec<u8> {
        self.pixel_buffer.to_vec()
    }
}

impl DetectionImage for image::DynamicImage {
    fn dimension(&self) -> (u32, u32) {
        self.dimensions()
    }

    fn pixel_buffer(&self) -> Vec<u8> {
        self.raw_pixels()
    }
}
