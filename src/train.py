import torch
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


class TinyCNN(nn.Module):
    """
    A minimal convolutional neural network for image classification.

    The network uses two convolutional layers with ReLU activations and
    max pooling followed by a small fully connected head. It is designed
    for small 28x28 grayscale images such as FashionMNIST.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the network."""
        return self.net(x)


def main() -> None:
    """
    Train the TinyCNN on FashionMNIST and save the model checkpoint and labels.

    This function downloads the FashionMNIST training dataset, trains the model for a
    few epochs, prints metrics for each epoch, and writes the trained model
    parameters and labels to the `artifacts` directory. If a GPU is
    available, it will be used automatically.
    """

    # Hyperparameters
    batch_size = 64
    epochs = 3
    lr = 1e-3

    # Transforms and dataset
    transform = tv.transforms.Compose([tv.transforms.ToTensor()])
    train_dataset = tv.datasets.FashionMNIST(
        root="/tmp/data",
        train=True,
        download=True,
        transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model, optimizer, and loss function
    model = TinyCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for ep in range(1, epochs + 1):
        total = 0
        correct = 0
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"epoch {ep}: loss={epoch_loss:.4f}, acc={epoch_acc:.3f}")

    # Save model and labels
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, artifacts_dir / "model.pt")
    with open(artifacts_dir / "labels.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_dataset.classes))
    print("saved -> artifacts/model.pt")


if __name__ == "__main__":
    main()
