import torch
from tqdm import tqdm
import os


def loss_function(y_pred, y_true, eps=1e-5):
    """Vector cross-entropy loss function."""
    h = y_true * torch.log(y_pred + eps)
    h = h.mean(-1).sum(-1)
    h = h.mean()
    return -h


def train_loop(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch_id, batch in tqdm(enumerate(dataloader), desc="Training Batches", total=len(dataloader)):
        x = batch["X"].to(device)
        x_preproc = model.annotate_batch_pre(
            x, {}
        )
        pred = model(x_preproc)
        loss = loss_function(pred, batch["y"].to(device))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Train Loss: {avg_loss:>7f} \n")

    return avg_loss


def _save_model(model, path: str):
    torch.save(model.state_dict(), path)


def eval_loop(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_id, batch in tqdm(enumerate(dataloader), desc="Evaluating Batches", total=len(dataloader)):
            x = batch["X"].to(device)
            x_preproc = model.annotate_batch_pre(
                x, {}
            )
            pred = model(x_preproc)
            loss = loss_function(pred, batch["y"].to(device))

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Eval Loss: {avg_loss:>7f} \n")

    return avg_loss


def train(model, optimizer, train_loader, dev_loader, epochs, device, save_directory=None):
    current_best_loss = float("inf")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        _ = train_loop(model, train_loader, optimizer, device)
        eval_loss = eval_loop(model, dev_loader, device)

        if current_best_loss > eval_loss:
            current_best_loss = eval_loss
            _save_model(model, os.path.join(save_directory, "best_model.pth"))
            print("Model saved!")

        _save_model(model, os.path.join(save_directory, "last_model.pth"))

    print("Training complete!")
