from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from models.transcoder import Transcoder


def train_transcoder(
    transcoder: "Transcoder",
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    l1_coefficient: float = 1e-3,
) -> None:
    """
    Train the transcoder model using the provided dataloader.
    
    Args:
        transcoder: An instance of the Transcoder model.
        dataloader: A PyTorch DataLoader providing (x_in, x_out) pairs for training.
        num_epochs: Number of epochs to train the model.
        learning_rate: Learning rate for the optimizer.
    """
    # Define an optimizer.
    optimizer = torch.optim.Adam(transcoder.parameters(), lr=learning_rate)
    device = next(transcoder.parameters()).device

    transcoder.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_x_in, batch_x_out in dataloader:
            batch_x_in = batch_x_in.to(device)
            batch_x_out = batch_x_out.to(device)

            optimizer.zero_grad()

            # Forward pass through transcoder
            x_out_pred, f = transcoder(batch_x_in)

            # Calculate Loss
            mse_loss = F.mse_loss(x_out_pred, batch_x_out)
            l1_loss = f.norm(p=1, dim=-1).mean()
            loss = mse_loss + l1_coefficient * l1_loss

            loss.backward()
            optimizer.step()

            # Normalize decoder weights after gradient step (Dictionary learning standard)
            with torch.no_grad():
                transcoder.W_dec.weight.data = F.normalize(transcoder.W_dec.weight.data, dim=0)

            epoch_loss += loss.item()
            num_batches += 1

        if epoch % 10 == 0:
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")