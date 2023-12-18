import torch
import torchvision
from torchvision.datasets import MNIST 
#######################################################################################################################################
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    model.validation_step = model.val_step
    model.validation_epoch_end = model.val_epoc_end
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
#######################################################################################################################################
def fit(epocs, lr, model, train_loader, val_loader, opt_fn=torch.optim.SGD):
    optimizer = opt_fn(model.parameters(),lr)
    history = []

    for epoc in range(epocs):
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoc_end(epoc, result)
        history.append(result)
    return history
#######################################################################################################################################