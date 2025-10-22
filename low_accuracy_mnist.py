import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# تعریف مدل ساده با تعداد نورون پایین
class SimpleDigitNet(nn.Module):
    def __init__(self):
        super(SimpleDigitNet, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 5)
        self.output = nn.Linear(5, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x))
        return self.output(x)


# آماده‌سازی داده‌ها
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_set_full = datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)
limited_train = Subset(train_set_full, range(6000))
test_set = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(limited_train, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1000)

# ساخت مدل و اجزای آموزش
net = SimpleDigitNet().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.005)


# تابع آموزش
def train_model(model, epochs):
    model.train()
    for ep in range(epochs):
        total_loss, correct = 0, 0
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = loss_function(out, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = out.argmax(1)
            correct += pred.eq(label).sum().item()
        acc = 100 * correct / len(train_loader.dataset)
        print(f"Epoch {ep+1} - Train Acc: {acc:.2f}%")


# تابع تست
def evaluate(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += pred.eq(y).sum().item()
    acc = 100 * correct / len(test_loader.dataset)
    print(f"Test Accuracy: {acc:.2f}%")


# تابع نمایش تصویری
def show_predictions(model, data_loader, num_samples=10):
    model.eval()
    images_shown = 0
    plt.figure(figsize=(15, 4))
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)

            for i in range(len(images)):
                if images_shown >= num_samples:
                    break
                img = images[i].cpu().squeeze().numpy()
                true_label = labels[i].item()
                pred_label = preds[i].item()
                plt.subplot(1, num_samples, images_shown + 1)
                plt.imshow(img, cmap="gray")
                plt.title(f"T:{true_label} | P:{pred_label}", fontsize=10)
                plt.axis("off")
                images_shown += 1

            if images_shown >= num_samples:
                break
    plt.tight_layout()
    plt.show()


# اجرا
train_model(net, epochs=10)
evaluate(net)
show_predictions(net, test_loader, num_samples=10)
