import torch

# %% Define the loss function
def quantile_loss(y_hat, y, quantiles):
    """
    y_hat: (batch_size, num_fcst, patch_len)
    y: (batch_size, num_fcst, patch_len)
    quantiles: list[float]
    """
    loss = torch.tensor(0.0).to(y.device)
    for i, q in enumerate(quantiles):
        error = y - y_hat
        loss += torch.max((q - 1) * error, q * error).mean()
    return loss / len(quantiles)


def rmse_loss(y_hat, y):
    """
    y_hat: (batch_size, num_fcst, patch_len)
    y: (batch_size, num_fcst, patch_len)
    """
    error = y - y_hat
    loss = torch.sqrt(torch.mean(error ** 2))
    return loss


def mae_loss(y_hat, y):
    """
    y: (batch_size, horizon)
    y_hat: (batch_size, horizon)
    """
    return torch.mean(torch.abs(y_hat - y))

def get_loss(args):
    if args.loss == 'quantile1':
        return lambda x, y: quantile_loss(x, y, [0.5])
    elif args.loss == 'quantile3':
        return lambda x, y: quantile_loss(x, y, [0.1, 0.5, 0.9])
    elif args.loss == 'quantile5':
        return lambda x, y: quantile_loss(x, y, [0.1, 0.3, 0.5, 0.7, 0.9])
    elif args.loss == 'rmse':
        return rmse_loss
    elif args.loss == 'mae':
        return mae_loss
    else:
        raise ValueError('loss not supported')