import torch.nn as nn
from typing import Tuple, Any
import torch
from torch.functional import Tensor

class RecurrentNeuralNetwork(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.hidden_size = hidden_size
    self.input2output = nn.Linear(input_size + hidden_size, output_size)
    self.input2hidden = nn.Linear(input_size + hidden_size, hidden_size)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, input, hidden) -> Tuple[Any, Any]:
    combined = torch.cat([input, hidden], 1)
    hidden = self.input2hidden(combined)
    output = self.input2output(combined)
    output = self.softmax(output)
    return output, hidden

  def get_hidden(self) -> Tensor:
    return torch.zeros(1, self.hidden_size)
