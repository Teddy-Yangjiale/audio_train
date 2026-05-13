import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. 加载数据
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 训练一个简单的随机森林分类器
print("正在训练模型...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 4. 评估效果
y_pred = model.predict(X_test)
print("\n训练结果报告：")
print(classification_report(y_test, y_pred, target_names=["yes", "no", "stop", "go"]))