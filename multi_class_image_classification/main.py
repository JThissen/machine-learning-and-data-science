# %%
import torch
from torch._C import device
from torch.functional import Tensor
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import ResNet
from torchvision import models
from torchsummary import summary
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as numpy
import pandas as pandas
import matplotlib.pyplot as pyplot
import os
from typing import List, Tuple

def main() -> None:
  train_dl, val_dl = prepare_data()
  train_validate_network(train_dl, val_dl)

def prepare_data(directory: str = "data", test_size: float = 0.25) -> Tuple[DataLoader, DataLoader]:
  path_data = f"./{directory}"

  if(not os.path.exists(path_data)):
    os.mkdir(f"./{directory}")

  train_set = torchvision.datasets.STL10(path_data, "train", download=True, transform=transforms.ToTensor())
  test_set = torchvision.datasets.STL10(path_data, "test", download=True, transform=transforms.ToTensor())
  sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
  indices = list(range(len(test_set)))
  y_test = [y for _,y in test_set]
  test_index = []
  val_index = []

  for i, j in sss.split(indices, y_test):
    test_index = i
    val_index = j
    break

  test_set = Subset(test_set, test_index)
  val_set = Subset(test_set, val_index)
  train_dl = DataLoader(train_set, batch_size=32, shuffle=True)
  val_dl = DataLoader(val_set, batch_size=32, shuffle=False)
  return train_dl, val_dl

def train_validate_network(train_dl: DataLoader, val_dl: DataLoader, directory: str = "data") -> None:
  network = models.resnet18(False) 
  network.fc = nn.Linear(network.fc.in_features, 10)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  network.to(device)
  loss_func = nn.CrossEntropyLoss(reduction="sum")
  optimizer = optim.Adam(network.parameters(), lr=0.001)
  lr = CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
  epochs = 20
  training_loss_list = []
  training_accuracy_list = []
  validation_loss_list = []
  validation_accuracy_list = []
  validation_loss = float("inf")

  for epoch in range(epochs):
    network.train()
    total_loss_training, accuracy_training = epoch_calculation(network, train_dl, loss_func, optimizer, device)
    training_loss_list.append(total_loss_training)
    training_accuracy_list.append(accuracy_training)
    print(f"[Training] Epoch: {epoch} | Total loss: {total_loss_training} | Accuracy: {accuracy_training}")
    
    network.eval()
    with torch.no_grad():
      total_loss_validation, accuracy_validation = epoch_calculation(network, train_dl, loss_func, optimizer, device, False)
      validation_loss_list.append(total_loss_validation)
      validation_accuracy_list.append(accuracy_validation)
      print(f"[Validation] Epoch: {epoch} | Total loss: {total_loss_validation} | Accuracy: {accuracy_validation}")

      if(total_loss_validation < validation_loss):
        validation_loss = total_loss_validation
        torch.save({"network_state_dict": network.state_dict()}, f"./{directory}/network_state_dict.pt")
    
    lr.step()
  
  plot_results(epochs, training_loss_list, training_accuracy_list, validation_loss_list, validation_accuracy_list)
  
def network_summary(network: ResNet, device: device) -> None:
  summary(network, input_size=(3, 224, 244), device=device.type) 

def plot_results(
  epochs: int, 
  training_loss_list: List[float],
  training_accuracy_list: List[float],
  validation_loss_list: List[float],
  validation_accuracy_list: List[float],
  background_color: Tuple[float, float, float] = (0.2, 0.2, 0.2)) -> None:

  figure, (ax1, ax2) = pyplot.subplots(1, 2)
  figure.set_figwidth(10)
  ax1.plot(list(range(1, epochs + 1)), numpy.array(training_loss_list), label="training")
  ax1.plot(list(range(1, epochs + 1)), numpy.array(validation_loss_list), label="validation")
  ax1.title.set_text("network loss per epoch")
  ax1.legend(loc="upper right")
  ax1.set_facecolor(background_color)
  ax2.plot(list(range(1, epochs + 1)), numpy.array(training_accuracy_list), label="training")
  ax2.plot(list(range(1, epochs + 1)), numpy.array(validation_accuracy_list), label="validation")
  ax2.title.set_text("network accuracy per epoch")
  ax2.set_facecolor(background_color)
  ax2.legend(loc="upper right")

def epoch_calculation(
  network: ResNet, 
  dataloader: DataLoader, 
  loss_func: CrossEntropyLoss,
  optimizer: Adam,
  device: device,
  gradient_calculation: bool = True) -> Tuple[int, float]:
  total_loss = 0
  total_correct = 0
  
  for i, j in dataloader:
    x: Tensor = i
    y: Tensor = j
    x = x.to(device)
    y = y.to(device)
    y_pred = network(x)
    loss = loss_func(y_pred, y)

    if(gradient_calculation):
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    total_loss += loss.item()
    maxIndex: Tensor = y_pred.argmax(dim=1, keepdim=True)
    total_correct += int(maxIndex.eq(y.view_as(maxIndex)).sum().item())

  accuracy = total_correct / len(dataloader.dataset)
  return total_loss, accuracy

def plot_image(x_data: Tensor, y_data: int):
  image = x_data.numpy()
  image_transposed = numpy.transpose(image, (1, 2, 0))
  pyplot.imshow(image_transposed)
  pyplot.title(f"label {y_data}")

if __name__ == "__main__":
  main()
# %%
