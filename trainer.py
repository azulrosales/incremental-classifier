import sys
import logging
import torch
import numpy as np
from model.aper_bn import Learner
from utils.data_manager import DataManager


def train(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
    )

    set_random()
    print_args(args)
    
    data_manager = DataManager(args["seed"])

    try:
        checkpoint = torch.load("model_checkpoint.pth")
        metadata = checkpoint["metadata"]
        session = metadata["session"]
        logging.info('Loaded metadata for session {}'.format("session"))
        print(metadata)
        model = Learner(args, metadata)
        model._network.load_state_dict(checkpoint["model_state"])
    except FileNotFoundError:
        logging.warning("No checkpoint found!")
        model = Learner(args)
        metadata = {"classes": [], "session": 0}
        session = 0
        
    model.incremental_train(data_manager)
    session += 1

    known_classes_names = [
        data_manager._class_mapping[class_id]
        for class_id in np.arange(model._known_classes, model._total_classes)
    ]

    if session > 0:
        metadata["classes"].extend(known_classes_names)
    else:
        metadata["classes"] = known_classes_names

    metadata["session"] = session

    logging.info("Saving the model after session {}".format(session))
    torch.save({"model_state": model._network.state_dict(), "metadata": metadata}, "model_checkpoint.pth")

    accuracies = model.eval_task(data_manager)
    model.after_task()
    
    logging.info("Accuracy: {}".format(accuracies["per_class"]))

def set_random():
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
