import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# 1. 数据加载和展示
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 下载并加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 展示数据集样本
def show_mnist_images(images, labels):
    images = images.numpy()
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        ax = axes[i]
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title(f'Label: {labels[i]}')
        ax.axis('off')
    plt.show()

# 读取一批数据
data_iter = iter(train_loader)
images, labels = next(data_iter)
show_mnist_images(images, labels)

# 2. 数据预处理
# 将数据集转换为二维数组（适用于逻辑回归）
def prepare_data(loader):
    data = []
    targets = []
    for images, labels in loader:
        images = images.view(images.size(0), -1)  # 将28x28的图像压平为784维向量
        data.append(images.numpy())
        targets.append(labels.numpy())
    data = np.vstack(data)
    targets = np.hstack(targets)
    return data, targets

X_train, y_train = prepare_data(train_loader)
X_test, y_test = prepare_data(test_loader)

# 3. 逻辑回归模型与5折交叉验证
print("开始5折交叉验证...")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
logistic_clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto', random_state=42)

# 交叉验证评分
cv_scores = cross_val_score(logistic_clf, X_train, y_train, cv=kfold, scoring='accuracy')
print("5折交叉验证准确率：", cv_scores)
print("平均准确率：", np.mean(cv_scores))

# 4. 模型训练与测试
print("开始训练逻辑回归模型...")
logistic_clf.fit(X_train, y_train)
y_pred = logistic_clf.predict(X_test)

# 5. 模型评估
print("测试集分类结果:")
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"测试集准确率: {accuracy:.4f}")
print("分类报告：\n", class_report)

# 混淆矩阵可视化
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
