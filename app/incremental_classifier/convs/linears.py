'''
Reference:
https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/cifar100-class-incremental/modified_linear.py
'''
import math
import torch
from torch import nn
from torch.nn import functional as F

class CosineLinear(nn.Module):
    '''
    Custom FC layer that computes Cosine Similarity between input features and weight vectors.
    '''
    def __init__(self, in_features, out_features, sigma=True):
        '''
        Initialize the CosineLinear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features (classes).
            sigma (bool): Whether to include a learnable scaling parameter.
        '''
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        '''
        Initialize the parameters of the layer.
        '''
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        '''
        Compute the forward pass for the layer.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, in_features).

        Returns:
            dict: A dictionary containing the key 'logits', with the cosine similarity 
                  logits as the value.
        '''
        # Normalize input and weight vectors for cosine similarity computation
        normalized_input = F.normalize(input, p=2, dim=1)  # Normalize along feature dimension
        normalized_weight = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity using normalized input and weights
        out = F.linear(normalized_input, normalized_weight)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}