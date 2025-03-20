import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CleanHazyDataset
from model import HybridDehazeNet

# 데이터 경로
clean_dir = r"C:\dev\bme\train\clean"
hazy_dir = r"C:\dev\bme\train\hazy"

# Transform 정의
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Dataset 및 DataLoader 생성
print("Loading dataset...")
dataset = CleanHazyDataset(clean_dir, hazy_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
print(f"Dataset loaded with {len(dataset)} samples.")

# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = HybridDehazeNet().to(device)
print("Model initialized.")

# 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
epochs = 10
print("Starting training...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    print(f"Epoch {epoch+1}/{epochs} started.")
    for batch_idx, (hazy_images, clean_images) in enumerate(data_loader):
        hazy_images, clean_images = hazy_images.to(device), clean_images.to(device)

        # Forward
        outputs = model(hazy_images)
        loss = criterion(outputs, clean_images)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # 각 배치마다 손실 출력
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(data_loader):
            print(f"Batch {batch_idx + 1}/{len(data_loader)}, Loss: {loss.item():.4f}")

    # 에포크 종료 시 손실 출력
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {epoch_loss / len(data_loader):.4f}")

# 모델 저장
torch.save(model.state_dict(), "dehazing_model.pth")
print("Model saved to 'dehazing_model.pth'.")
