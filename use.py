import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader # stores samples and labels, iterates thru dataset
from torchvision import datasets, transforms # preexisting datasets/transforms
from torchvision.transforms import ToTensor # turns pictures into tensors
import matplotlib.pyplot as plt # displays the pictures and labels
from learn import NeuralNetwork

model = NeuralNetwork()
model.load_state_dict(torch.load("model_weights.pth", weights_only = True))
model.eval()

dataset = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

input = dataset[torch.randint(len(dataset), size=(1,)).item()][0] # get a random picture from the dataset

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# predict the category of the image
logits = model(input)
predict_prob = nn.Softmax(dim = 1)(logits)
y_predict = predict_prob.argmax(1)
print(f"Predicted: {labels_map[y_predict.item()]}")

# plot out the chosen image
plt.imshow(input.squeeze(), cmap = "gray")
plt.show()