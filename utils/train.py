import torch

### Train Loop ###
# Performs 1 epoch of training for the model.
# Inputs: 
# dataloader - the DataLoader containing the training data.
# model - The model to be trained.
# loss_fn - The Loss function to be evaluated for back propagation.
# Optimizer - The optimizer used to backprop. (Usually Adam)
# Device - the device that the model is on. (cuda or cpu)
def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    # collect dataset size
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    # Loop over all batches
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        # Perform Forward Pass
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Back Prop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Display ongoing Statistics
        if batch % 100 == 0:
            current = batch * batch_size + len(X)
            print(f'loss: {loss.item():>7f} [{current:>5d}/{size:>5d}]')

    return loss

### Test Loop ###
# Evaluates the model on the test or validation dataset
# Inputs: 
# dataloader - the DataLoader containing the data to be evaluated.
# model - The trained model
# loss_fn - The Loss function that sets the evaluation criteria.
# Device - the device that the model is on. (cuda or cpu)
def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    # Collect Dataset Size
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0,0

    with torch.no_grad():
        # Loop over all batches
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            # Forward Pass
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Compute Statistics
        test_loss /= num_batches
        correct /= size

        print(f'Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')

        return test_loss


### Train ###
# Trains a model and evaluates performance on a dev/validation set.
# Inputs: 
# train_dl - the DataLoader containing the training data.
# model - The model to be trained.
# loss_fn - The Loss function to be evaluated for back propagation.
# Optimizer - The optimizer used to backprop. (Usually Adam)
# Device - the device that the model is on. (cuda or cpu)
# Epochs - The number of epochs to train for. (Default: 10)
# val_dl - The dataloader containing the validation/dev 
def train(train_dl, model, loss_fn, optimizer, device, epochs = 10, val_dl = None):
    save_weights_only = False
    train_loss_curve = []
    test_loss_curve = []
    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train_loss = train_loop(train_dl, model, loss_fn, optimizer, device)
        train_loss_curve.append(train_loss)
        if val_dl is not None:
            test_loss = test_loop(val_dl, model, loss_fn, device)
            test_loss_curve.append(test_loss)

    if save_weights_only:
        print('Saving full model...')
        torch.save(model, 'models/saved_model.pth')
    else:
        print('Saving model weights only...')
        torch.save(model.state_dict(), 'models/saved_model_weights.pth')
    print('Done!')
    return train_loss_curve, test_loss_curve