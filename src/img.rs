use image::GenericImageView;

#[derive(Default, Clone)]
pub struct GenericImage {
    width: u32,
    height: u32,
    pixel_buffer: Vec<u8>,
}

impl GenericImage {
    pub fn new(width: u32, height: u32, pixel_buffer: Vec<u8>) -> Self {
        GenericImage {
            width,
            height,
            pixel_buffer,
        }
    }
}

impl DetectionImage for GenericImage {
    fn dimension(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn pixel_buffer(&self) -> Vec<u8> {
        self.pixel_buffer.iter().cloned().collect()
    }
}

pub trait DetectionImage {
    fn dimension(&self) -> (u32, u32);
    fn pixel_buffer(&self) -> Vec<u8>;
}

impl DetectionImage for image::DynamicImage {
    fn dimension(&self) -> (u32, u32) {
        self.dimensions()
    }

    fn pixel_buffer(&self) -> Vec<u8> {
        self.raw_pixels()
    }
}
