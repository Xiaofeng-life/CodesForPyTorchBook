import torch
from llcnn_net import LLCNN
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    # 构建模型并加载LLCNN权重
    device = torch.device("cpu")
    llcnn = LLCNN().to(device)
    llcnn.load_state_dict(torch.load("results/pth/20.pth", map_location="cpu"))
    llcnn.eval()
    # 用于验证增强效果的图像
    image_path = "gamma_dataset/val_dark/1355.jpg"

    with torch.no_grad():
        low_light_image = Image.open(image_path)
        low_light_image = low_light_image.resize((256, 256))
        low_light_image = TF.to_tensor(low_light_image).to(device).unsqueeze(0)
        enhanced_image = llcnn(low_light_image)
        enhanced_image = torch.clamp(enhanced_image, min=0, max=1)
        # 将低光照图像与正常光照图像拼接并保存
        result = torch.cat((low_light_image, enhanced_image), dim=3)[0]
        result = result * 255
        result = result.cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8)
        result = result[5:250,5:505,:]
        plt.imshow(result)
        plt.savefig("demo.png", dpi=500, bbox_inchs="tight")
        plt.show()
