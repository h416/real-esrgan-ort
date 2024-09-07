# real-esrgan-ort


This is a [ort](https://github.com/pykeio/ort) ( Rust binding for ONNX Runtime ) implementation of [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), it depends on [real-esrgan-onnx](https://github.com/instant-high/real-esrgan-onnx) model.
 
## how to build and run

### install rust
https://www.rust-lang.org/tools/install

### clone the repository

    git clone https://github.com/h416/real-esrgan-ort.git
    cd real-esrgan-ort

### download model

    wget https://github.com/instant-high/real-esrgan-onnx/releases/download/RealESRGAN-ONNX/RealEsrganONNX.zip
    unzip RealEsrganONNX.zip

### build

    cargo build --release

### run

    ./target/release/real-esrgan-ort RealESRGAN_x2.onnx input.jpg output.png
