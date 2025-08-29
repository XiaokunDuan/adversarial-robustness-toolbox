import torch
import torchvision.transforms as transforms

class ColorMNIST(torch.utils.data.Dataset):
    """
    创建一个彩色版的MNIST，作为目标域。
    数字为绿色，背景为蓝色。
    """
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
    
    def __getitem__(self, index):
        image, label = self.mnist_dataset[index]
        image = image.squeeze()
        
        # 转换为3通道RGB
        image_rgb = torch.zeros(3, 28, 28)
        
        # 将数字部分变为绿色，背景变为蓝色
        digit_mask = image > 0 # Find where the digit is (binary mask)
        image_rgb[1, digit_mask] = image[digit_mask] # Green channel
        image_rgb[2, ~digit_mask] = 0.8 # Blue channel
        
        # 归一化到 [-1, 1]
        final_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image_rgb)
        
        return final_image, label

    def __len__(self):
        return len(self.mnist_dataset)