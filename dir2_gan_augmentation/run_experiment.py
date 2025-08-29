import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import os
import matplotlib.pyplot as plt

from models import Generator, Discriminator, ClassifierCNN

# --- 全局设置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
LATENT_DIM = 100

# --- GAN 训练函数 ---
def train_gan(generator, discriminator, loader, epochs=20): # GAN需要更多轮次
    generator.train()
    discriminator.train()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    adversarial_loss = nn.BCELoss()
    
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(loader):
            # 真实图像
            real_imgs = imgs.to(DEVICE)
            valid = torch.ones(imgs.size(0), 1, device=DEVICE)
            fake = torch.zeros(imgs.size(0), 1, device=DEVICE)

            # --- 训练生成器 ---
            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), LATENT_DIM, device=DEVICE)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()
            
            # --- 训练判别器 ---
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        print(f"GAN Epoch {epoch+1}/{epochs} [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
        
        # 保存一些生成的图片样本
        if (epoch + 1) % 5 == 0:
            save_image(gen_imgs.data[:25], os.path.join(RESULTS_DIR, f"gan_epoch_{epoch+1}.png"), nrow=5, normalize=True)

# --- 分类器训练函数 ---
def train_classifier(model, loader, epochs=5):
    model.train()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Classifier Epoch {epoch+1}/{epochs} done.")

# --- 分类器测试函数 ---
def test_classifier(model, loader, test_name):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy on {test_name}: {accuracy:.2f}%")
    return accuracy

# --- 主执行流程 ---
def main():
    # 1. 训练GAN
    print("--- Training GAN ---")
    gan_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    gan_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=gan_transform)
    gan_loader = DataLoader(gan_dataset, batch_size=128, shuffle=True)
    
    generator = Generator(latent_dim).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    train_gan(generator, discriminator, gan_loader)
    torch.save(generator.state_dict(), os.path.join(RESULTS_DIR, "gan_generator.pth"))
    print("GAN Generator saved.")

    # 2. 准备分类器的数据集
    cls_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    original_train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=cls_transform)
    original_train_loader = DataLoader(original_train_set, batch_size=128, shuffle=True)

    # 3. 创建增强数据集
    print("\n--- Creating Augmented Dataset ---")
    num_generated = 60000 # 生成与原始数据同样多的样本
    with torch.no_grad():
        z = torch.randn(num_generated, LATENT_DIM, device=DEVICE)
        generated_images = generator(z).cpu()
        generated_images = (generated_images * 0.5) + 0.5 # 逆归一化到 [0, 1]
        generated_images = (generated_images - 0.1307) / 0.3081 # 重新归一化以匹配分类器
        generated_labels = torch.randint(0, 10, (num_generated,))
    
    gan_dataset = TensorDataset(generated_images, generated_labels)
    augmented_dataset = ConcatDataset([original_train_set, gan_dataset])
    augmented_loader = DataLoader(augmented_dataset, batch_size=128, shuffle=True)
    print(f"Augmented dataset size: {len(augmented_dataset)}")

    # 4. 训练标准分类器
    print("\n--- Training Standard Classifier ---")
    standard_classifier = ClassifierCNN().to(DEVICE)
    train_classifier(standard_classifier, original_train_loader)
    torch.save(standard_classifier.state_dict(), os.path.join(RESULTS_DIR, "standard_classifier.pth"))

    # 5. 训练鲁棒分类器
    print("\n--- Training Robust Classifier on Augmented Data ---")
    robust_classifier = ClassifierCNN().to(DEVICE)
    train_classifier(robust_classifier, augmented_loader)
    torch.save(robust_classifier.state_dict(), os.path.join(RESULTS_DIR, "robust_classifier.pth"))

    # 6. 对比测试
    print("\n--- Final Robustness Comparison ---")
    rotated_transform = transforms.Compose([
        transforms.RandomRotation(degrees=(-45, 45)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    rotated_test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=rotated_transform)
    rotated_test_loader = DataLoader(rotated_test_set, batch_size=1000, shuffle=False)

    print("--- Standard Classifier ---")
    test_classifier(standard_classifier, rotated_test_loader, "Rotated Test Data")
    
    print("\n--- Robust (GAN-Augmented) Classifier ---")
    test_classifier(robust_classifier, rotated_test_loader, "Rotated Test Data")

if __name__ == "__main__":
    main()