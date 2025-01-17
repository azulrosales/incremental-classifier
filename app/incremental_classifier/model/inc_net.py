import copy
import torch
from torch import nn
from ..convs.linears import CosineLinear
from ..convs.resnet import resnet18


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_convnet(args):
    '''
    Creates and returns a ResNet18 model instance.

    Args:
        args: Arguments required for initializing the ResNet18 model.

    Returns:
        torch.nn.Module: The ResNet18 model instance set to evaluation mode.
    '''
    model = resnet18(args=args)
    return model.eval()

class BaseNet(nn.Module):
    def __init__(self, args):
        super(BaseNet, self).__init__()
        print('This is for the BaseNet initialization.')
        self.convnet = get_convnet(args)
        print('After BaseNet initialization.')
        self.fc = None

    @property
    def feature_dim(self):
        '''
        Returns the output feature dimension of the convolutional network.
        '''
        return self.convnet.out_dim

    def extract_vector(self, x):
        '''
        Extracts feature vectors from the input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Extracted feature vectors.
        '''
        return self.convnet(x)["features"]

    def forward(self, x):
        '''
        Forward pass for the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: A dictionary containing the logits and extracted features.
        '''
        x = self.convnet(x)
        out = self.fc(x["features"])

        out.update(x)

        return out

    def update_fc(self):
        pass

    def generate_fc(self):
        pass

    def copy(self):
        '''
        Creates a deep copy of the current network instance.
        '''
        return copy.deepcopy(self)

    def freeze(self):
        '''
        Freezes the network parameters to make them non-trainable.
        '''
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

class SimpleCosineIncrementalNet(BaseNet):
    '''
    A simple incremental network that uses cosine similarity for classification.
    '''
    def __init__(self, args):
        super().__init__(args)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        '''
        Updates the FC layer to accommodate new classes.

        Args:
            nb_classes (int): Total number of classes after the update.
            nextperiod_initialization (torch.Tensor or None): Optional tensor 
                for initializing weights for the new classes.
        '''
        print('SimpleCosineIncrementalNet: Update FC')
        fc = self.generate_fc(self.feature_dim, nb_classes).to(device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        '''
        Generates a new fully connected layer.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.

        Returns:
            CosineLinear: A CosineLinear layer instance.
        '''
        fc = CosineLinear(in_dim, out_dim)
        return fc


class MultiBranchCosineIncrementalNet(BaseNet):
    '''
    An incremental network that supports multi-branch architectures for classification.
    '''
    def __init__(self, args):
        super().__init__(args)
        print('Clear the convnet in MultiBranchCosineIncrementalNet, since we are using self.convnets with dual branches')
        self.convnet = torch.nn.Identity()
        for param in self.convnet.parameters():
            param.requires_grad = False

        self.convnets = nn.ModuleList()
        self.args = args

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        '''
        Updates the fully connected layer for the multi-branch network.

        Args:
            nb_classes (int): Total number of classes after the update.
            nextperiod_initialization (torch.Tensor or None): Optional tensor 
                for initializing weights for the new classes.
        '''
        print('MultiBranchCosineIncrementalNet: Update FC')
        fc = self.generate_fc(self._feature_dim, nb_classes).to(device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self._feature_dim).to(device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        '''
        Generates a new fully connected layer.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.

        Returns:
            CosineLinear: A CosineLinear layer instance.
        '''
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        '''
        Forward pass for the multi-branch network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: A dictionary containing logits and combined features.
        '''
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        out = self.fc(features)
        out.update({"features": features})
        return out
        
    def construct_dual_branch_network(self, tuned_model, nb_classes):
        '''
        Constructs a dual-branch architecture.

        Args:
            tuned_model (BaseNet): The adapted tuned model.
            nb_classes (int): Total number of classes.
        '''
        
        # Add the pretrained model as the first branch
        self.convnets.append(get_convnet(self.args)) 
        # Add the tuned model as the second branch
        self.convnets.append(tuned_model.convnet)

        # Update feature dimension and fully connected layer
        self._feature_dim = self.convnets[0].out_dim * len(self.convnets) 
        self.fc = self.generate_fc(self._feature_dim, nb_classes) 
        