from typing import List, Tuple
import glob
import unicodedata
import torch
from torch.functional import Tensor

class Utils:
  @staticmethod
  def get_files(path: str) ->List[str]:
    return glob.glob(path)

  @staticmethod
  def unicodeToAscii(s: str, all_letters: str) -> str:
    return ''.join(
      c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn'
      and c in all_letters
    )

  @staticmethod
  def read_lines(filename, all_letters: str) -> List[str]:
    lines_list = open(filename, encoding="utf-8").read().split("\n")
    lines = []
    for i in lines_list:
      lines.append(Utils.unicodeToAscii(i.strip(), all_letters))
    return lines

  @staticmethod
  def letter_to_index(letter: str, all_letters: str) -> int:
    return all_letters.find(letter)

  @staticmethod
  def letter_to_tensor(letter: str, all_letters: str) -> Tensor:
    tensor = torch.zeros(1, len(all_letters))
    tensor[0][Utils.letter_to_index(letter, all_letters)] = 1
    return tensor

  @staticmethod
  def line_to_tensor(line: str, all_letters: str) -> Tensor:
    tensor = torch.zeros(len(line), 1, len(all_letters))
    for index, letter in enumerate(line):
      tensor[index][0][Utils.letter_to_index(letter, all_letters)] = 1
    return tensor

  @staticmethod
  def category_from_output(output: Tensor, all_categories: List[str]) -> Tuple[str, int]:
    _, index = output.topk(1)
    return all_categories[index.item()], index.item()
