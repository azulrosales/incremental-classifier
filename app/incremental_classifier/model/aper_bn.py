import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from .inc_net import SimpleCosineIncrementalNet, MultiBranchCosineIncrementalNet
from ..utils.toolkit import tensor2numpy, accuracy, generate_confusion_matrix, st_log

# Tune the model (with forward BN) at first session, and then conduct simple shot.

num_workers = 0

class Learner(object):
    '''
    Implements the APER BN incremental learning strategy with a focus on batch normalization (BN) adaptation.
    Supports training, evaluation, and inference using a dual-branch network.
    '''
    def __init__(self, args={}, metadata=None):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.topk = 2
        self._known_classes = len(metadata["classes"])
        self._session = metadata["session"]
        self._create_network()
        self.batch_size = args.get('batch_size', 128)
        self.tune_epochs = args.get('tune_epochs', 1)

    @property
    def feature_dim(self):
        """
        Returns the feature dimensionality of the network.
        """
        return self._network.feature_dim

    def _train(self):
        """
        Perform training. Handles the initial training and further incremental sessions.
        """
        self._network.to(self._device)

        if self._session == 0:
            self._init_train(self.train_loader)
            self.construct_dual_branch_network()
        else:
            pass

        self.replace_fc(self.train_loader_for_protonet, self._network)

    def _init_train(self, train_loader):
        """
        Conduct the initial training phase.

        Args:
            train_loader (DataLoader): Data loader for the training dataset.
        """
        print('APER BN: Initial Training')

        # Reset the running statistics of the BN layers
        self.clear_running_mean()

        # Adapt to the current data via forward passing
        prog_bar = tqdm(range(self.tune_epochs), desc='Adapting to new data')
        with torch.no_grad():
            for epoch in prog_bar:
                self._network.train()
                for i, (_, inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    logits = self._network(inputs)["logits"]
                    del logits

    def replace_fc(self, train_loader, model):
        """
        Replace the FC layers with prototypes for each class.

        Args:
            train_loader (DataLoader): Data loader for feature extraction.
            model (torch.nn.Module): Current model.

        Returns:
            Updated model with replaced FC layers.
        """
        model = model.eval()

        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model(data)['features']
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        st_log('Replacing FC Layers...')
        class_list = np.unique(self.train_dataset.labels)
        for class_index in class_list:
            st_log(f"- Replacing: {class_index}")
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto
        return model

    def incremental_train(self, data_manager, total_classes):
        """
        Perform incremental training for a new session.

        Args:
            data_manager: Manages datasets for training and testing.
            total_classes (int): Total number of classes encountered so far.
        """
        print('APER BN: Incremental Train')
        self.data_manager = data_manager
        
        # Load training data
        train_dataset = data_manager.get_dataset(source="train", mode="train")
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self._curr_classes = len(np.unique(train_dataset.labels))
        self._total_classes = total_classes

        # Update network FC layers for new classes
        self._network.update_fc(self._total_classes)
        
        # Load test data
        test_dataset = data_manager.get_dataset(source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        # Prepare protonet training data
        train_dataset_for_protonet = data_manager.get_dataset(source="train", mode="test")
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        self._train()

    def construct_dual_branch_network(self):
        '''
        Build a dual-branch incremental network.
        '''
        print('APER BN: Constructing MultiBranchCosineIncrementalNet')
        network = MultiBranchCosineIncrementalNet(self.args)
        network.construct_dual_branch_network(self._network, self._curr_classes)
        self._network = network.to(self._device)

    def _create_network(self):
        """
        Create the appropriate network for the session (initial or incremental).
        """
        if self._session == 0:
            self._network = SimpleCosineIncrementalNet(self.args)
        else:
            self._network = MultiBranchCosineIncrementalNet(self.args)
            from ..convs.resnet import resnet18
            from ..convs.linears import CosineLinear
            self._network.convnets.append(resnet18(args=self.args))
            self._network.convnets.append(resnet18(args=self.args))
            self._network._feature_dim = self._network.convnets[0].out_dim * len(self._network.convnets)
            self._network.fc = CosineLinear(self._network._feature_dim, self._known_classes)
            self._network.to(self._device)

    def clear_running_mean(self):
        """
        Reset the running mean and variance of batch normalization layers.
        """
        print('APER BN: Cleaning Running Mean')
        # Record the index of running mean and variance
        model_dict = self._network.state_dict()
        running_dict = {}
        for e in model_dict:
            if 'running' in e:
                key_name = '.'.join(e.split('.')[1:-1])
                if key_name in running_dict:
                    continue
                else:
                    running_dict[key_name] = {}
                # Find the position of BN modules
                component = self._network.convnet
                for att in key_name.split('.'):
                    if att.isdigit():
                        component = component[int(att)]
                    else:
                        component = getattr(component, att)

                running_dict[key_name]['mean'] = component.running_mean
                running_dict[key_name]['var'] = component.running_var
                running_dict[key_name]['nbt'] = component.num_batches_tracked

                component.running_mean = component.running_mean * 0
                component.running_var = component.running_var * 0
                component.num_batches_tracked = component.num_batches_tracked * 0

    def after_task(self):
        """
        Update known classes after completing training.
        """
        self._known_classes = self._total_classes

    def _evaluate(self, y_pred, y_true, data_manager):
        ret = {}
        per_class = accuracy(y_pred.T[0], y_true, data_manager)
        ret["per_class"] = per_class
        ret["top1"] = per_class["Total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        generate_confusion_matrix(y_true, y_pred, data_manager)

        return ret

    def eval_task(self, data_manager):
        """
        Evaluate the model on the current session.

        Args:
            data_manager: Data manager for obtaining test data.

        Returns:
            dict: Evaluation metrics including accuracy.
        """
        y_pred, y_true = self._eval_cnn(self.test_loader)
        accuracy = self._evaluate(y_pred, y_true, data_manager)

        return accuracy

    def _compute_accuracy(self, model, loader):
        """
        Compute accuracy for the given model and data loader.

        Args:
            model (torch.nn.Module): Model to evaluate.
            loader (DataLoader): Data loader for evaluation.

        Returns:
            float: Accuracy in percentage.
        """
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        """
        Evaluates the current network on a given data loader and returns predictions and true labels.

        Args:
            loader (torch.utils.data.DataLoader): The data loader providing input data and corresponding labels.

        Returns:
            tuple:
                - y_pred (numpy.ndarray): Predicted top-k class indices for all inputs, of shape [N, topk].
                - y_true (numpy.ndarray): True class labels for all inputs, of shape [N].
        """
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"] # Forward pass
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # Top-k predictions
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    
    def _infer(self, img_path):
        """
        Performs inference on a single image and returns the predicted top-k class indices.

        Args:
            img_path (str): Path to the image file to be classified.

        Returns:
            numpy.ndarray: Predicted top-k class indices, of shape [topk].
        """
        from PIL import Image
        from torchvision import transforms
        from ..utils.toolkit import build_transform
        
        # Load and preprocess the image
        img = Image.open(img_path).convert("RGB")
        transform = transforms.Compose(build_transform(is_train=False))
        img = transform(img)

        # Perform inference
        self._network.eval() 
        img = img.unsqueeze(0).to(self._device)  
        with torch.no_grad():
            outputs = self._network(img)["logits"] # Forward pass
        predicts = torch.topk(
            outputs, k=self.topk, dim=1, largest=True, sorted=True
        )[1] # Top-k predictions
        
        return predicts.cpu().numpy().flatten()