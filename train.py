import argparse
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from collections import OrderedDict
import model_utils  # Custom module containing model definition and training functions


def get_input_args():
    parser = argparse.ArgumentParser(description="Train a neural network")
    parser.add_argument('data_directory', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units1', type=int, default=4096, help='Number of hidden units 1')
    parser.add_argument('--hidden_units2', type=int, default=1024, help='Number of hidden units 2')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--cat_names', dest="cat_names", default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()

def main():
    args = get_input_args()
    
    # Load data
    train_loader, valid_loader, test_loader, train_data, valid_data, test_data = model_utils.load_data(args.data_directory)
    
    # Build model
    model, criterion, optimizer, scheduler = model_utils.build_model(args.arch, args.dropout, args.learning_rate, args.hidden_units1, args.hidden_units2,total_classes=102)
    
    #Check_gpu
    device= model_utils.check_gpu(args.gpu)
    
    #Train the model
    model_utils.train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, args.epochs)

    #Save model

    model_utils.save_checkpoint(model, train_data, optimizer, args.epochs, args.arch, args.save_dir)
    
if __name__ == '__main__':
    main()
