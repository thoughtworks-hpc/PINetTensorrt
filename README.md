## key points estimation and point instance segmentation approach for lane detection

- Paper : key points estimation and point instance segmentation approach for lane detection
- Paper Link : https://arxiv.org/abs/2002.06604
- Author : Yeongmin Ko, Jiwon Jun, Donghwuy Ko, Moongu Jeon (Gwanju Institute of Science and Technology)


- This repository is TensorRT implement of [PINet](github.com/koyeongmin/PINet)


## Dependency

- TensorRT 6.0
- OpenCV

## Convert

you can convert Pytorch weights file to onnx file, follow as:

- insert this code at end of agent.py :
  
```python
    def export_onnx(self, input_image, filename):
        torch_out = torch.onnx.export(self.lane_detection_network, input_image, filename, verbose=True)
```

- run this code to convert weights file to onnx, please use pytorch 1.0.1

```python
    import torch
    import agent

    batch_size = 1
    input_shape = (3, 256, 512)
    dummy_input = torch.randn(batch_size, *input_shape, device='cuda')
    lane_agent = agent.Agent()
    lane_agent.load_weights(640, "tensor(0.2298)")
    lane_agent.cuda()
    lane_agent.evaluate_mode()
    lane_agent.export_onnx(dummy_input, "pinet.onnx")
```

## Run
- run this program with image directory

```shell
    ./PINetTensorrt --datadir=<path of your test images> 
```

- or run this program with default images
  
```shell
    ./PINetTensorrt
```

## Test

### Object
- Pytorch implement of [PINet](github.com/koyeongmin/PINet)
- Tensorrt C++ implement of PINet


### Purpose

- Tensorrt performance under X86 architecture
- Tensorrt performance under Xavier


### Dataset

data source：tusimple dataset 0531 directory

image format：jpg

image size：1280 x 720

image channels：RGB

count of images：14300

disk space size： 3GB


---
### X86 Computer

OS：ubuntu 18.04

CPU：AMD Ryzen 7 3700X 8-Core Processor

CPU Frequency: 3600 mhz

ram：32GB  3200mhz

video card：Nvidia Titan

vram： 6G

disk：Seagate cool fish 7200rpm



### Xavier

OS：ubuntu 18.04

CPU：ARMv8 Processor rev 0 (v8l)

CPU Frequency: 2036 mhz

ram：16GB


### result

##### explain

- end to end：elapsed time of read image， inference，post processing，draw lane line result to image
- execute： elapsed time of copy host ram to device vram，inference exectute， copy device vram to host ram
- totally end to end : elapsed time of dataset test, sum of end to end
- totally execute： elapsed time of dataset test, sum of execute

```
end to end = totally end to end / count of image in dataset 
execute    = totally execute    / count of image in dataset
```


##### X86 && Pytorch Implement

| NO. | totally end to end | end to end(ms) | totally execute(s) | execute(ms) |
| ---- | ------------------ | ---------------- | ------------------- | ------------- |
| 1    | 39m54.79s      | 167.46           | 229.92              | 16.07         |
| 2    | 38m28.37s      | 161.42           | 222.29              | 15.54         |
| 3    | 38m04.18s      | 159.73           | 224.73              | 15.71         |
| 4    | 37m40.54s      | 158.08           | 218.91              | 15.30         |
| 5    | 38m05.84s      | 159.84           | 223.84              | 15.65         |
| average | 38m26.74s      | 161.31           | 233.94              | 15.65         |



##### X86 && Tensorrt C++ Implement

| NO. | totally end to end(s) | end to end(ms) | totally execute(s) | execute(ms) |
| ---- | ---------------------- | ---------------- | ------------------- | ------------- |
| 1    | 335.970                | 23.494           | 152.362             | 10.654        |
| 2    | 346.983                | 24.264           | 154.873             | 10.83         |
| 3    | 338.014                | 23.637           | 153.296             | 10.72         |
| 4    | 342.812                | 23.972           | 154.606             | 10.811        |
| 5    | 343.489                | 24.02            | 154.693             | 10.817        |
| average | 341.45                 | 23.88            | 153.966             | 10.77         |


##### Xavier && Tensorrt C++ Implement

| NO. | totally end to end(s) | end to end(ms) | totally execute(s) | execute(ms) |
| ---- | ---------------------- | ---------------- | ------------------- | ------------- |
| 1    | 709.816                | 49.637           | 289.398             | 20.237        |
| 2    | 652.201                | 45.608           | 287.493             | 20.104        |
| 3    | 651.780                | 45.578           | 290.308             | 20.301        |
| 4    | 650.099                | 45.461           | 287.789             | 20.125        |
| 5    | 657.376                | 45.97            | 287.569             | 20.109        |
| average | 664.254                | 46.45            | 288.51              | 20.175        |


##### Result
- elapsed time of inference execute under x86 architecture, Tensorrt C++ implement faster 1.5 times than Pytorch implement
- elapsed time of end to end under x86 architecture, Tensorrt C++ implement faster 10 times than Pytorch implement
- elapsed time of inference execute under Xavier, x86 architecture faster 2 times， takes 20 ms on average 