import torch

# %% Define the loss function
def quantile_loss(y_hat, y, quantiles):
    """
    y_hat: (batch_size, horizon, quantiles)
    y: (batch_size, horizon)
    quantiles: list[float]
    """
    if len(y_hat.shape) == 2:
        y_hat = y_hat.unsqueeze(-1)
    assert y_hat.shape[-1] == len(quantiles), f"{y_hat.shape[-1]} != {len(quantiles)}"
    loss = torch.tensor(0.0).to(y.device)
    for i, q in enumerate(quantiles):
        error = y - y_hat[:, :, i]
        loss += torch.max((q - 1) * error, q * error).mean()
    return loss / len(quantiles)


def rmse_loss(y, y_hat):
    """
    y: (batch_size, horizon)
    y_hat: (batch_size, horizon)
    """
    return torch.sqrt(torch.mean((y_hat - y) ** 2))


def mae_loss(y, y_hat):
    """
    y: (batch_size, horizon)
    y_hat: (batch_size, horizon)
    """
    return torch.mean(torch.abs(y_hat - y))