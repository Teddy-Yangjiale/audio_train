import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # 必须放在最前面！
import torch
import torch.fx.experimental.sym_node
if not hasattr(torch.fx.experimental.sym_node, 'DynamicInt'):
    torch.fx.experimental.sym_node.DynamicInt = int
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 准备数据
X = np.load("X_features.npy") # 现在的维度应该是 (2000, 39)
y = np.load("y_labels.npy")

# 标准化数据（深度学习对数值范围非常敏感，这一步极其重要！）
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# 2. 定义 MLP 神经网络
class AudioMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(AudioMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2), # 防止过拟合
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# 初始化模型 (输入 39 维，输出 4 类)
model = AudioMLP(input_size=39, num_classes=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 训练循环
print("开始深度学习训练...")
for epoch in range(50): # 跑 50 轮
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")

# 4. 评估
model.eval()
with torch.no_grad():
    y_pred_logits = model(X_test)
    y_pred = torch.argmax(y_pred_logits, dim=1)
    accuracy = (y_pred == y_test).float().mean()
    print(f"\n深度学习模型准确率: {accuracy.item() * 100:.2f}%")