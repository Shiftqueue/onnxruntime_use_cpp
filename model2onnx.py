import torch
import numpy as np
import onnx
from models.unet import GRFBUNet

def to_onnx(model_unet):
    with torch.no_grad():
        img_unet = torch.randn(1,3,640,640)
        outputs_unet = model_unet(img_unet)

        # 导出ONNX文件
        torch.onnx.export(
            model_unet,
            img_unet,
            'grfb_unet.onnx',
            opset_version=11,
            input_names=['input'],
            output_names=['output']
        )

        # prediction = outputs_unet['out'].argmax(1)
        
    return 


def main():
    # 加载unet模型
    model_unet = GRFBUNet(in_channels=3, num_classes=2, base_c=32)
    model_unet.load_state_dict(torch.load('./weights/grfb-unet.pth', map_location='cpu')['model'])
    model_unet.eval() 
    to_onnx(model_unet)


if __name__ == "__main__":
    main()
