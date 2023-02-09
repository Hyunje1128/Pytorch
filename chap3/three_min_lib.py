import torch

class three_min:
    def print_tensor(tensor):
        print("Size:", tensor.size())
        print("Shape:", tensor.shape)
        print("랭크(차원):", tensor.ndimension())