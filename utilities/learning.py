from torch.optim.lr_scheduler import LinearLR, ExponentialLR
import torch
import matplotlib.pyplot as plt
import tqdm

#export
def valley(lrs:list, losses:list, num_it:int) -> (float, tuple):
    "Suggests a learning rate from the longest valley and returns its index"
    n = len(losses)

    max_start, max_end = 0,0

    # find the longest valley
    lds = [1]*n

    for i in range(1,n):
        for j in range(0,i):
            if (losses[i] < losses[j]) and (lds[i] < lds[j] + 1):
                lds[i] = lds[j] + 1
            if lds[max_end] < lds[i]:
                max_end = i
                max_start = max_end - lds[max_end]

    sections = (max_end - max_start) / 3
    idx = max_start + int(sections) + int(sections/2)

    return lrs[idx]


def LRfinder(model, optimizer, loss_fn, dataloader, start_lr=1e-8, end_lr=10.0,
             beta=0.98, device='cpu', step_mode='exp', num_iter=100, plot=True):
    # save model to load back after fitting

    model.train()
    losses = []
    lrs = []
    lr = start_lr
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.0
    best_loss = 0.0
    batch_num = 0

    # Initialize the proper learning rate policy
    if step_mode.lower() == "exp":
        gamma = (end_lr / start_lr) ** (1 / num_iter)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
    elif step_mode.lower() == "linear":
        end_factor = end_lr / start_lr
        scheduler = LinearLR(optimizer,start_factor=1, end_factor=end_factor, total_iters=num_iter)
    else:
        raise ValueError("Expected step_mode to be one of 'exp' or 'linear'")
    while lr < end_lr:
        for inputs, targets in dataloader:
            batch_num += 1
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** batch_num)
            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                break
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # Store the values
            losses.append(smoothed_loss)
            lrs.append(scheduler.get_last_lr()[0])
            # Do the SGD step
            loss.backward()
            optimizer.step()
            # Update the learning rate
            scheduler.step()

    suggested_lr = valley(lrs, losses, num_iter)

    if plot:
        plt.plot(lrs, losses)
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        # set the x scale to log
        plt.xscale('log')
        # mark the suggested learning rate
        plt.axvline(suggested_lr, color='r', linestyle='--')
        plt.show()

    return suggested_lr