import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader # stores samples and labels, iterates thru dataset
from torchvision import datasets, transforms # preexisting datasets/transforms
from torchvision.transforms import ToTensor # turns pictures into tensors
import matplotlib.pyplot as plt # displays the pictures and labels

# define neural network structure
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28**2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, input): # aka going forward thru the nn
        input = self.flatten(input)
        return self.linear_relu_stack(input)
    
# define hyperparameters
learning_rate = 10**-3
batch_size = 64
epochs = 20

# define how to do training and testing
def training_loop(dataloader, model, loss_function, optimizer):
    model.train() # set model to training mode
    for batch, (input, expected) in enumerate(dataloader):
        # compute prediction and loss
        prediction = model(input)
        loss = loss_function(prediction, expected)

        # do back prop
        loss.backward() # find how much to change each weight/bias
        optimizer.step() # apply the changes to each weight/bias
        optimizer.zero_grad() # zero out the gradients so they don't build up (they don't clear by default)

        # diagnostic
        if batch % 100 == 0:
            current = batch * batch_size + len(input)
            # print loss as width 7 float, print [current/total] as width 5 decimal(int)
            print(f"Loss: {loss.item():>7f} [{current:>5d}/{len(dataloader.dataset):>5d}]")
def testing_loop(dataloader, model, loss_function):
    model.eval() # set model to testing mode

    test_loss = 0
    correct = 0

    with torch.no_grad(): # don't compute any gradients in testing mode
        for input, expected in dataloader:
            predicted = model(input)
            test_loss += loss_function(predicted, expected).item()
            # if the prediction is correct, cast to float tensor, sum, and convert to number
            correct += (predicted.argmax(1) == expected).type(torch.float).sum().item()

    test_loss /= len(dataloader) # divide by the number of batches
    correct /= len(dataloader.dataset) # divide by the total number of test cases

    print(f"Test Error:\nAccuracy: {(correct * 100):>.2f}%, Average Loss: {test_loss:>8f}\n")

def main():
    # get training and testing datasets
    training_data = datasets.FashionMNIST(
        root = "data",
        train = True,
        download = True,
        transform = ToTensor()
    )
    testing_data = datasets.FashionMNIST(
        root = "data",
        train = False,
        download = True,
        transform = ToTensor()
    )
    training_dataloader = DataLoader(training_data, batch_size = batch_size)
    testing_dataloader = DataLoader(testing_data, batch_size = batch_size)

    # check if gpu available
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # initialize model, loss function, optimizer
    model = NeuralNetwork()#.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

    # do the actual learning
    for curr_epoch in range(epochs):
        print(f"Epoch: {curr_epoch+1} --------------------")
        training_loop(training_dataloader, model, loss_function, optimizer)
        testing_loop(testing_dataloader, model, loss_function)

    torch.save(model.state_dict(), "model_weights.pth")

if __name__ == "__main__":
    main()