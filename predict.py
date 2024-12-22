import argparse
import torch
from torchvision import models
#from PIL import Image
import json
#import numpy as np
import model_utils

def get_input_args():
    parser = argparse.ArgumentParser(description='Predict flower name from an image along with class probability.')
    # Required arguments
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('checkpoint', type=str, default="checkpoint.pth", help='Path to the model checkpoint')
    
    # Optional arguments
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to a JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    return parser.parse_args()
    
def main():
    args = get_input_args()
    
    # Load the model
    
    model, optimizer, class_to_idx, epochs = model_utils.load_checkpoint(args.checkpoint)
    # Set device
    #Check_gpu
    device= model_utils.check_gpu(args.gpu)
    # Predict
    #top_probs, top_classes = model_utils.predict(args.image_path, model, args.top_k, args.category_names, device)
    top_probs, top_classes = model_utils.predict(args.image_path, model, device, args.top_k)
  
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    # Print results
    print([cat_to_name[i] for i in top_classes])
    print(f"The flower is {cat_to_name[top_classes[0]]}")

if __name__ == '__main__':
    main()