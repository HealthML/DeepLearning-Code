import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# ----------------------------------------------------------------------------------------
# Data and Visualizations
def create_dataset(n_samples=600, n_classes=3, seed=42):
    np.random.seed(seed)
    X, y = [], []
    for i in range(n_classes):
        center = np.random.randn(2) * 5
        points = np.random.randn(n_samples // n_classes, 2) + center
        X.append(points)
        y.append(np.ones(n_samples // n_classes, dtype=int) * i)
    X = np.vstack(X)
    y = np.hstack(y)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def plot_dataset(X, y, title="Dataset"):
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k")
    plt.title(title)
    plt.show()


def get_decision_grid(model, X, device="cpu"):
    """Return grid coordinates and predictions for decision surface plotting"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    with torch.no_grad():
        Z = model(torch.FloatTensor(grid).to(device)).argmax(dim=1).cpu().numpy()
    return xx, yy, Z.reshape(xx.shape)

# ----------------------------------------------------------------------------------------
# Models

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=3):
        super().__init__()
        self.linear = None
        # Hint: Use nn.Linear for fully connected layer

    def forward(self, x):
        return self.linear(x)

class SimpleNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=3):
        super().__init__()        
        self.net = None
        # Hint: 
            # See nn.Sequential for building feedforward networks
            # Use nn.ReLU() for activation functions
            # Use nn.Linear for fully connected layers

    def forward(self, x):
        return self.net(x)

class DeepNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=3):
        super().__init__()
        self.net = None
        # Hint: 
            # Build a deeper network with more layers and activations
            # If you have built SimpleNN correctly, this should be similar but with more layers
            # Use nn.Sequential, nn.Linear, and nn.ReLU as needed

    def forward(self, x):
        return self.net(x)



# ----------------------------------------------------------------------------------------
# Training Utils

def train(model, X_train, y_train, lr=0.01, epochs=200, device="cpu"):
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

def evaluate(model, X_test, y_test, device="cpu"):
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.LongTensor(y_test).to(device)
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
    acc = (preds == y_test).float().mean().item()
    return acc





def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset
    X, y = create_dataset()
    plot_dataset(X, y, "Generated Dataset")

    # Train/test split
    n = len(X)
    split = int(0.7 * n)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Normalize (based on train stats)
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    num_classes = len(np.unique(y))

    # ------------------------
    # Train 3 models
    # ------------------------
    models = [
        ("Logistic Regression", LogisticRegressionModel(output_dim=num_classes), 0.05, 200),
        ("Simple NN", SimpleNN(output_dim=num_classes), 0.01, 300),
        ("Deep NN", DeepNN(output_dim=num_classes), 0.01, 400)
    ]

    results = []

    for name, model, lr, epochs in models:
        model = model.to(device)
        print(f"\nTraining {name}...")
        losses = train(model, X_train, y_train, lr=lr, epochs=epochs, device=device)
        acc = evaluate(model, X_test, y_test, device)
        print(f"Test Accuracy ({name}): {acc:.3f}")
        results.append((name, model, losses, acc))

    # ------------------------
    # Combined Visualization
    # ------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, model, _, acc) in zip(axes, results):
        xx, yy, Z = get_decision_grid(model, X_test, device)
        ax.contourf(xx, yy, Z, alpha=0.4, cmap="viridis")
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k", cmap="viridis")
        ax.set_title(f"{name}\nTest Acc = {acc:.2f}")
    plt.suptitle("Decision Boundaries on Test Set", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # You are given the skeleton code above which includes helpers for:
        # Dataset generation and visualization
        # Training and evaluation loops
        # Plotting decision boundaries and losses
    # Your task is to implement the missing parts of the models above
        # LogisticRegressionModel
        # SimpleNN
        # DeepNN
    # Happy coding!

    main()
