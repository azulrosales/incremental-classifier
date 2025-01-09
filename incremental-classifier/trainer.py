import sys
import logging
import torch
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
    
    model_path = "./checkpoint/model_checkpoint.pth"

    try:
        checkpoint = torch.load(model_path)
        metadata = checkpoint["metadata"]
        logging.info('Loaded metadata for session {}'.format(metadata["session"]))
        print(metadata)
        model = Learner(args, metadata)
        model._network.load_state_dict(checkpoint["model_state"])
    except FileNotFoundError:
        logging.warning("No checkpoint found!")
        metadata = {"classes": [], "session": 0}
        model = Learner(args, metadata)
        
    data_manager = DataManager(known_classes=metadata["classes"])

    new_classes_names = data_manager._class_names
    metadata["classes"].extend(cls for cls in new_classes_names if cls not in metadata["classes"])
    
    model.incremental_train(data_manager, total_classes=len(metadata["classes"]))
    metadata["session"] += 1

    logging.info("Saving the model after session {}".format(metadata["session"]))
    torch.save({"model_state": model._network.state_dict(), "metadata": metadata}, model_path)

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
