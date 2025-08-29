import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os

from models import SimpleCNN
from attacks import fgsm_attack

# --- 全局设置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- 数据加载 ---
def get_data_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)
    return train_loader, test_loader

# --- 训练函数 ---
def train(model, loader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} done.")

def adversarial_train(model, loader, optimizer, criterion, epsilon, epochs=5):
    model.train()
    for epoch in range(epochs):
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # 生成对抗样本进行训练
            perturbed_data = fgsm_attack(model, criterion, data, target, epsilon)
            
            optimizer.zero_grad()
            output = model(perturbed_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Adversarial Training Epoch {epoch+1}/{epochs} done.")

# --- 测试函数 ---
def test(model, loader, test_name, epsilon=0):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        if epsilon > 0:
            data = fgsm_attack(model, criterion, data, target, epsilon)

        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
    accuracy = 100 * correct / total
    print(f"Accuracy on {test_name}: {accuracy:.2f}%")
    return accuracy

# --- 可视化函数 ---
def visualize_attack_comparison(standard_model, robust_model, loader, epsilon):
    # 从测试集中获取一个样本
    data, target = next(iter(loader))
    image, label = data[0:1].to(DEVICE), target[0:1].to(DEVICE) # 取一个样本

    # 标准模型攻击
    perturbed_std = fgsm_attack(standard_model, nn.CrossEntropyLoss(), image.clone(), label, epsilon)
    pred_std_clean = standard_model(image).max(1)[1].item()
    pred_std_adv = standard_model(perturbed_std).max(1)[1].item()

    # 鲁棒模型攻击
    perturbed_robust = fgsm_attack(robust_model, nn.CrossEntropyLoss(), image.clone(), label, epsilon)
    pred_robust_clean = robust_model(image).max(1)[1].item()
    pred_robust_adv = robust_model(perturbed_robust).max(1)[1].item()
    
    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Adversarial Attack Comparison (Epsilon = {epsilon})", fontsize=16)

    # 原始图像
    for i in range(2):
        axes[i, 0].imshow(image.cpu().detach().numpy().squeeze(), cmap="gray")
        axes[i, 0].set_title(f"Original Image (Label: {label.item()})")
        axes[i, 0].axis('off')

    # 标准模型行
    axes[0, 0].set_ylabel("Standard Model", fontsize=14)
    axes[0, 1].imshow((perturbed_std - image).cpu().detach().numpy().squeeze(), cmap="gray")
    axes[0, 1].set_title(f"Perturbation\nClean Pred: {pred_std_clean}")
    axes[0, 1].axis('off')
    axes[0, 2].imshow(perturbed_std.cpu().detach().numpy().squeeze(), cmap="gray")
    axes[0, 2].set_title(f"Adversarial Image\nAdv Pred: {pred_std_adv}")
    axes[0, 2].axis('off')
    
    # 鲁棒模型行
    axes[1, 0].set_ylabel("Robust Model", fontsize=14)
    axes[1, 1].imshow((perturbed_robust - image).cpu().detach().numpy().squeeze(), cmap="gray")
    axes[1, 1].set_title(f"Perturbation\nClean Pred: {pred_robust_clean}")
    axes[1, 1].axis('off')
    axes[1, 2].imshow(perturbed_robust.cpu().detach().numpy().squeeze(), cmap="gray")
    axes[1, 2].set_title(f"Adversarial Image\nAdv Pred: {pred_robust_adv}")
    axes[1, 2].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(RESULTS_DIR, "attack_comparison.png"))
    print(f"Comparison visualization saved to {os.path.join(RESULTS_DIR, 'attack_comparison.png')}")

# --- 主执行流程 ---
def main():
    train_loader, test_loader = get_data_loaders()
    criterion = nn.CrossEntropyLoss()
    epsilon = 0.25

    # 1. 训练标准模型
    print("--- Training Standard Model ---")
    standard_model = SimpleCNN().to(DEVICE)
    optimizer_std = optim.Adam(standard_model.parameters())
    train(standard_model, train_loader, optimizer_std, criterion)
    torch.save(standard_model.state_dict(), os.path.join(RESULTS_DIR, "standard_model.pth"))
    print("Standard model saved.")

    # 2. 训练鲁棒模型
    print("\n--- Training Robust Model (Adversarial Training) ---")
    robust_model = SimpleCNN().to(DEVICE)
    optimizer_robust = optim.Adam(robust_model.parameters())
    adversarial_train(robust_model, train_loader, optimizer_robust, criterion, epsilon)
    torch.save(robust_model.state_dict(), os.path.join(RESULTS_DIR, "robust_model.pth"))
    print("Robust model saved.")

    # 3. 对比测试
    print("\n--- Final Performance Comparison ---")
    print(f"Attack Epsilon = {epsilon}\n")
    print("--- Standard Model ---")
    test(standard_model, test_loader, "Clean Data", epsilon=0)
    test(standard_model, test_loader, "Adversarial Data", epsilon=epsilon)
    
    print("\n--- Robust Model ---")
    test(robust_model, test_loader, "Clean Data", epsilon=0)
    test(robust_model, test_loader, "Adversarial Data", epsilon=epsilon)

    # 4. 可视化
    print("\n--- Generating Visualization ---")
    visualize_attack_comparison(standard_model, robust_model, test_loader, epsilon)

if __name__ == "__main__":
    main()