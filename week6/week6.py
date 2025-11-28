import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # add 2 convolutional blocks
        # The first block: conv(1, 6, 5) -> relu -> max-pool(2x2)
        # The second block: conv(6, 16, 5) -> relu -> max-pool(2x2)
        
        self.conv = nn.Sequential(
            # add here
        )

        # add three linear layers
        # The first linear layer: Fully connected(SIZE, 120) -> relu
            # SIZE depends on the output of conv layers
            # Check the dimensions after conv layers to compute SIZE (it's a good exercise for you :))
        # The second linear layer: Fully connected(120, 84) -> relu
        # The second linear layer: Fully connected(84, 10)
        
        self.fc = nn.Sequential(
            # add here
            )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)



def visualize_conv_features(model, image):

    model.eval()
    device = next(model.parameters()).device

    x = image.unsqueeze(0).to(device)  # [1, 1, 28, 28]

    plt.figure(figsize=(3, 3))
    plt.imshow(image.squeeze().cpu(), cmap="gray")
    plt.title("Input")
    plt.axis("off")
    plt.show()

    conv_layer_num = 0
    for layer in model.conv:
        x = layer(x)

        if isinstance(layer, nn.Conv2d):
            conv_layer_num += 1
            feats = x.squeeze(0).detach().cpu()  # [C, H, W]
            num_maps = feats.size(0)

            cols = min(8, num_maps)
            rows = (num_maps + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
            axes = axes.flatten()

            # Plot the feature maps
            for i in range(num_maps):
                # add here
                pass


            fig.suptitle(f"Conv layer {conv_layer_num}: {num_maps} feature maps")
            plt.tight_layout()
            plt.show()


def validate(model, data_loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total

    model.train()
    return avg_loss, accuracy


def main():
    batch_size = 32 # feel free to change this
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

        val_loss, val_acc = validate(model, test_loader, loss_fn, device)
        print(
            f"Epoch {epoch + 1}: "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    torch.save(model.state_dict(), "lenet_mnist.pth")

    test_iter = iter(test_loader)
    images, labels = next(test_iter)
    sample_img = images[0].to(device)

    visualize_conv_features(model, sample_img)


if __name__ == "__main__":
    main()
