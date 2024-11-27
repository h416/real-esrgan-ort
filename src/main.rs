use image::{imageops::FilterType, DynamicImage, GenericImageView};

use ort::execution_providers::CUDAExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionOutputs};
use ort::tensor::PrimitiveTensorElementType;
use ort::value::Tensor;

use half::f16;

fn load_image(path: &str, filter: FilterType) -> DynamicImage {
    let original_img = image::open(path).unwrap();

    let mut width = original_img.width();
    let mut height = original_img.height();
    if width % 2 == 0 && height % 2 == 0 {
        original_img
    } else {
        if width % 2 != 0 {
            width -= 1;
        }
        if height % 2 != 0 {
            height -= 1;
        }

        let img = original_img.resize_exact(width, height, filter);
        img
    }
}

fn main() -> ort::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        let command = &args[0];
        println!("usage {} RealESRGAN_x2.onnx input.jpg output.png", &command);
        return Ok(());
    }
    let model_path = &args[1];
    let input_path = &args[2];
    let output_path = &args[3];
    let is_fp16 = model_path.ends_with("_fp16.onnx");
    // println!("is_fp16 {}", is_fp16);
    if is_fp16 {
        convert::<f16>(model_path, input_path, output_path)
    } else {
        convert::<f32>(model_path, input_path, output_path)
    }
}

fn pixel_to_value<T: num_traits::float::Float + num_traits::cast::FromPrimitive>(x: u8) -> T {
    let value_f32 = x as f32 / 255.0_f32;
    let result: T = T::from_f32(value_f32).unwrap();
    result
}

fn image_to_vec<T: num_traits::float::Float + num_traits::cast::FromPrimitive>(
    img: &DynamicImage,
) -> Vec<T> {
    let img_width = img.width();
    let img_height = img.height();
    let mut vec = Vec::<T>::with_capacity(3 * img_height as usize * img_width as usize);
    for col in 0..3 {
        for y in 0..img_height {
            for x in 0..img_width {
                let pixel = img.get_pixel(x, y);
                let value = pixel_to_value(pixel[col]);
                vec.push(value);
            }
        }
    }
    vec
}

fn vec_to_image<T: num_traits::float::Float>(
    output_vec: &Vec<T>,
    img_width: u32,
    img_height: u32,
) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    let scale2 = output_vec.len() as u32 / img_width / img_height / 3;
    let scale = f64::sqrt(scale2 as f64) as u32;
    // println!("scale {}", scale);
    let output_width = img_width * scale;
    let output_height = img_height * scale;
    // println!("{}x{}", output_width, output_height);

    let output_size = (output_width * output_height) as usize;
    let output_img = image::ImageBuffer::from_fn(output_width, output_height, |x, y| {
        let i = (x + y * output_width) as usize;
        let rf = output_vec[output_size * 0 + i].to_f32().unwrap();
        let gf = output_vec[output_size * 1 + i].to_f32().unwrap();
        let bf = output_vec[output_size * 2 + i].to_f32().unwrap();
        let r = (rf.clamp(0.0, 1.0) * 255.0_f32.round()) as u8;
        let g = (gf.clamp(0.0, 1.0) * 255.0_f32.round()) as u8;
        let b = (bf.clamp(0.0, 1.0) * 255.0_f32.round()) as u8;
        image::Rgb([r, g, b])
    });
    output_img
}
fn predict<T: PrimitiveTensorElementType + std::fmt::Debug + std::clone::Clone + 'static>(
    model_path: &str,
    input_vec: &Vec<T>,
    img_width: usize,
    img_height: usize,
) -> ort::Result<Vec<T>> {
    let num_cpus = num_cpus::get();

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(num_cpus)?
        .commit_from_file(model_path)?;

    let input_tensor = Tensor::from_array((
        [1_usize, 3_usize, img_height, img_width],
        // input_vec.into_boxed_slice(),
        input_vec.clone().into_boxed_slice(),
    ))?;

    let outputs: SessionOutputs = model.run(inputs![input_tensor]?)?;

    let output = &outputs[0];
    let output_tensor = output.try_extract_tensor::<T>()?;
    let output_vec: Vec<T> = output_tensor.map(|x| x.clone()).into_iter().collect();
    Ok(output_vec)
}

fn convert<
    T: num_traits::float::Float
        + num_traits::cast::FromPrimitive
        + PrimitiveTensorElementType
        + std::fmt::Debug
        + std::clone::Clone
        + 'static,
>(
    model_path: &str,
    input_path: &str,
    output_path: &str,
) -> ort::Result<()> {
    // let input_resize_type = FilterType::Nearest;
    // let input_resize_type = FilterType::CatmullRom;
    // let input_resize_type = FilterType::Gaussian;
    let input_resize_type = FilterType::Lanczos3;

    if input_path == output_path {
        println!("input and output is same");
        return Ok(());
    }

    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;

    let img = load_image(input_path, input_resize_type);
    let width = img.width();
    let height = img.height();

    let input_vec: Vec<T> = image_to_vec(&img);
    drop(img);

    let output_vec = predict::<T>(model_path, &input_vec, width as usize, height as usize)?;
    drop(input_vec);

    let output_img = vec_to_image::<T>(&output_vec, width, height);
    drop(output_vec);

    output_img.save(output_path).unwrap();
    Ok(())
}
