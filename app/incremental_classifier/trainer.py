import sys
import logging
import torch
import pandas as pd
import streamlit as st
from .model.aper_bn import Learner
from .utils.data_manager import DataManager
from .utils.toolkit import st_log


def train(args):
    '''
    Main function to train or incrementally train a model using specified arguments.
    '''

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
    )

    set_random()
    print_args(args)
    
    MODEL_PATH = "../checkpoint/model_checkpoint.pth"

    st.write("#####")

    try:
        checkpoint = torch.load(MODEL_PATH)
        metadata = checkpoint["metadata"]
        st_log(f"Loaded metadata for session {metadata['session']}:")
        st_log(f"- Classes: {metadata['classes']}")
        # Initialize model with metadata
        model = Learner(args, metadata)
        model._network.load_state_dict(checkpoint["model_state"])
    except FileNotFoundError:
        # Handle missing checkpoint scenario
        if args.get('mode') == 'Incremental Train':
            st.warning('😿 No checkpoint found!')
            return False
        else:
            logging.warning("No checkpoint found!")
            metadata = {"classes": [], "session": 0}
            model = Learner(args, metadata)
        
    data_manager = DataManager(known_classes=metadata["classes"])

    # Update metadata with new classes
    new_classes_names = data_manager._class_names
    metadata["classes"].extend(cls for cls in new_classes_names if cls not in metadata["classes"])
    
    # Perform incremental training
    model.incremental_train(data_manager, total_classes=len(metadata["classes"]))
    metadata["session"] += 1

    # Save the model and updated metadata
    st_log(f"Saving the model after session {metadata["session"]}...")
    torch.save({"model_state": model._network.state_dict(), "metadata": metadata}, MODEL_PATH)

    # Evaluate the model on the task
    accuracies = model.eval_task(data_manager)
    model.after_task()
    
    # Log and save class-wise accuracies
    st_log("Class-wise Accuracies:")
    df = pd.DataFrame(list(accuracies["per_class"].items()), columns=["Class Name", "Accuracy"])
    df.set_index("Class Name", inplace=True)
    df.to_csv('../checkpoint/test_accuracy.csv')
    st.dataframe(df, use_container_width=True)

    return True

def set_random():
    '''
    Sets random seed for reproducibility across PyTorch and CUDA operations.
    '''
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def print_args(args):
    '''
    Print given arguments.
    '''
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
