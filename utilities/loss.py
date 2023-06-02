import torch
import numpy as np

# %% Define the loss function
def quantile_loss(y_hat, y, quantiles):
    """
    y_hat: (batch_size, num_fcst, patch_len, num_quantiles)
    y: (batch_size, num_fcst, patch_len)
    quantiles: list[float]
    """
    loss = torch.tensor(0.0).to(y.device)
    for i, q in enumerate(quantiles):
        error = y - y_hat[:, :, :, i]
        loss += torch.max((q - 1) * error, q * error).mean()
    return loss / len(quantiles)


def rmse_loss(y_hat, y):
    """
    y_hat: (batch_size, num_fcst, patch_len, *)
    y: (batch_size, num_fcst, patch_len)
    """
    # todo: deal with stride != patch_len, num_fcst = y_hat.shape[1]
    if len(y_hat.shape) == 4:
        y_hat = y_hat[:, :, :, 0]
    error = y - y_hat
    loss = torch.sqrt(torch.mean(error ** 2))
    return loss


def mae_loss(y_hat, y):
    """
    y: (batch_size, horizon)
    y_hat: (batch_size, horizon)
    """
    if len(y_hat.shape) == 4:
        y_hat = y_hat[:, :, :, 0]
    return torch.mean(torch.abs(y_hat - y))

def get_loss(args):
    if args.loss == 'quantile1':
        return lambda x, y: quantile_loss(x, y, [0.5])
    elif args.loss == 'quantile3':
        return lambda x, y: quantile_loss(x, y, [0.5, 0.1, 0.9])
    elif args.loss == 'quantile5':
        return lambda x, y: quantile_loss(x, y, [0.5, 0.3, 0.7, 0.1, 0.9])
    elif args.loss == 'rmse':
        return rmse_loss
    elif args.loss == 'mae':
        return mae_loss
    elif args.loss == 'gaussian_nll':
        loss = torch.nn.GaussianNLLLoss()
        return lambda y_hat, y: loss(y_hat[:, :, :, 0], y, y_hat[:, :, :, 1]**2)
    else:
        raise ValueError('loss not supported')


def evaluate(model, test_loader, device='cpu'):
    """Evaluate the model on the test set.
    Returns
    -------
    test_mae, test_rmse
    """
    test_mae = 0
    test_mse = 0
    model.eval()
    len_data = len(test_loader.dataset)
    with torch.no_grad():
        for x,y in test_loader:
            len_batch = len(x[0])
            x = [xx.to(device) for xx in x]
            y = y.to(device)
            y_hat = model(x)
            test_mae += mae_loss(y_hat, y).item() * len_batch / len_data
            test_mse += rmse_loss(y_hat, y).item()**2 * len_batch / len_data
        test_rmse = test_mse**0.5
    return test_mae, test_rmse
