import copy
import torch
from torch import nn
from convs.linears import CosineLinear
from convs.resnet import resnet18


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_convnet(args):
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
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])

        out.update(x)

        return out

    def update_fc(self):
        pass

    def generate_fc(self):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args):
        super().__init__(args)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
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
        fc = CosineLinear(in_dim, out_dim)
        return fc

class MultiBranchCosineIncrementalNet(BaseNet):
    def __init__(self, args):
        super().__init__(args)
        
        # No need the convnet.
        print('Clear the convnet in MultiBranchCosineIncrementalNet, since we are using self.convnets with dual branches')
        self.convnet = torch.nn.Identity()
        for param in self.convnet.parameters():
            param.requires_grad = False

        self.convnets = nn.ModuleList()
        self.args = args
        self.modeltype = 'cnn'

    def update_fc(self, nb_classes, nextperiod_initialization=None):
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
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        out = self.fc(features)
        out.update({"features": features})
        return out
        
    def construct_dual_branch_network(self, tuned_model, nb_classes):
        self.convnets.append(get_convnet(self.args)) # the pretrained model itself

        self.convnets.append(tuned_model.convnet) # adapted tuned model
    
        self._feature_dim = self.convnets[0].out_dim * len(self.convnets) 
        self.fc=self.generate_fc(self._feature_dim, nb_classes) 
        