import torch

def fgsm_attack(model, loss_fn, image, label, epsilon):
    """
    实现快速梯度符号法 (FGSM) 攻击。
    """
    # 将image设置为需要计算梯度的张量
    image.requires_grad = True
    
    # 前向传播
    output = model(image)
    
    # 计算损失
    loss = loss_fn(output, label)
    
    # 反向传播，计算梯度
    model.zero_grad()
    loss.backward()
    
    # 采集梯度数据
    gradient = image.grad.data
    
    # 找到梯度的符号
    sign_gradient = gradient.sign()
    
    # 创建对抗样本
    perturbed_image = image + epsilon * sign_gradient
    
    # 注意：MNIST数据集加载时已经归一化，所以我们不需要裁剪到[0,1]
    # 如果是未归一化的图像，需要加: perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image