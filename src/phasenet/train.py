import torch
from tqdm import tqdm


def loss_function(y_pred, y_true, eps=1e-5):
    """Vector cross-entropy loss function."""
    h = y_true * torch.log(y_pred + eps)
    h = h.mean(-1).sum(-1)
    h = h.mean()
    return -h


def train_loop(model, dataloader, optimizer, device):
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


def eval_loop(model, dataloader, device):
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


def train(model, optimizer, train_loader, dev_loader, epochs, device):

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(model, train_loader, optimizer, device)
        eval_loop(model, dev_loader, device)

    print("Training complete!")
