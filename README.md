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
