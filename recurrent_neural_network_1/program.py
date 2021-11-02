import torch
import torch.nn as nn
import string
import os
import time
import random
from torch.functional import Tensor
from utils import Utils
from recurrent_neural_network import RecurrentNeuralNetwork
from typing import List, Tuple, Any

class Program():
  def __init__(self, learning_rate: float = 0.005, iterations: int = 100000, hidden_length: int = 128):
    self.category_lines = {}
    self.all_categories = []
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.all_letters = string.ascii_letters + ".,;"
    self.learning_rate = learning_rate
    self.iterations = iterations
    self.hidden_length = hidden_length
    self.loss_func = nn.NLLLoss()

  def run(self, train_model: bool = True):
    files = Utils.get_files("./data/names/*.txt")

    for i in files:
      name = os.path.splitext(os.path.basename(i))[0]
      self.all_categories.append(name)
      self.category_lines[name] = Utils.read_lines(i, self.all_letters)

    total_loss = 0
    correct_count = 0
    losses: List[str] = []
    rnn = RecurrentNeuralNetwork(len(self.all_letters), self.hidden_length, len(self.all_categories))
    start = time.time()

    if(train_model):
      for i in range(self.iterations):
        category, _, category_tensor, line_tensor = self.random_example()

        if((line_tensor.size()[0]) == 0):
          continue

        output, loss = self.train(rnn, category_tensor, line_tensor)
        total_loss += loss
        losses.append(loss)
        result, _ = Utils.category_from_output(output, self.all_categories)
        correct = result == category

        if(correct is True):
          correct_count += 1
          
        print(f"iter: {i}, correct: {correct}")

      print(f"correct percentage: {(correct_count / self.iterations) * 100.0}")
      print(f"elapsed time: {time.time() - start}")
      torch.save(rnn.state_dict(), "./network.pt")
    else:
      rnn.load_state_dict(torch.load("./network.pt"))
      rnn.eval()
      self.predict(rnn, "Thissen", 3)

  def train(self, rnn: RecurrentNeuralNetwork, category_tensor: Tensor, line_tensor: Tensor) -> Tuple[Any, float]:
    hidden = rnn.get_hidden()
    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
      output, hidden = rnn(line_tensor[i], hidden)

    loss = self.loss_func(output, category_tensor)
    loss.backward()

    for i in rnn.parameters():
      i.data.add_(i.grad.data, alpha=-self.learning_rate)

    return output, loss.item()

  def predict(self, rnn: RecurrentNeuralNetwork, line: str, predictions_amount: int = 5) -> None:
    with torch.no_grad():
      hidden = rnn.get_hidden()
      line_tensor = Utils.line_to_tensor(line, self.all_letters)
      for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
      _, indices = output.topk(predictions_amount, 1, True)

      for i in range(predictions_amount):
        print(f"prediction: {self.all_categories[indices[0][i].item()]}")

  def random_example(self) -> Tuple[str, str, Tensor, Tensor]:
    random_category = self.all_categories[random.randint(0, len(self.all_categories)-1)]
    random_word = self.category_lines[random_category][random.randint(0, len(self.category_lines[random_category])-1)]
    category_tensor = torch.tensor([self.all_categories.index(random_category)], dtype=torch.long)
    line_tensor = Utils.line_to_tensor(random_word, self.all_letters)
    return random_category, random_word, category_tensor, line_tensor
