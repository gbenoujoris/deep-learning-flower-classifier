
# -*- coding: utf-8 -*-                                                                           
# PROGRAMMER: GBENOU Joris Martial Adechina

import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.optim import lr_scheduler
import json
from collections import OrderedDict
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F 


def load_data(data_dir='./flowers/'):
    """
    Arguments : Main path toward all the data(train, loader, test)
    Return : transformed data and data loader
    """
    #data_dir = 'C:/Users/Joris/Documents/Udacity_AWS/aipnd-project-master' # Can remplace the lin
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms_train =  transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]) 


    data_transforms_valid = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])
                                     ])

    data_transforms_test = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]) 

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms_train)
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms_valid)
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms_test)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return train_loader, valid_loader, test_loader, train_data, valid_data, test_data


def save_checkpoint(model, train_data, optimizer, epochs, arch, filepath='checkpoint.pth'):
    """
    Saves the current state of a trained model, optimizer, and other relevant 
    information to a checkpoint file. 
    
    Parameters:
      1. model - the trained model to be saved .
      2. train_data - the training dataset, used to retrieve the class-to-index mapping.
      3. optimizer - the optimizer used during training (e.g., Adam) whose state will be saved.
      4. epochs - the number of epochs the model has been trained for.
      5. filepath - the location where the checkpoint will be saved (default is 'checkpoint.pth').

    Returns:
      None - (a confirmation message.)
    """
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier if hasattr(model, 'classifier') else model.fc,  # Add classifier or fc based on model
        'epochs': epochs,
        'arch': arch  # Save the model architecture in the checkpoint
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at: {filepath}")
    
    
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_units1, hidden_units2, output_size, dropout):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_units1)
        self.fc2 = nn.Linear(hidden_units1, hidden_units2)
        self.fc3 = nn.Linear(hidden_units2, output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

def build_model(arch, dropout, learning_rate, hidden_units1, hidden_units2, total_classes=102):
    """
    Builds a deep learning model based on the specified architecture and parameters. 
    Parameters:
      1. arch (str) - the architecture of the pretrained model to be used. Options include:
         'vgg16', 'vgg13', 'densenet121', or 'resnet50'.
      2. dropout (float) - the dropout probability to be applied in the custom classifier.
      3. learning_rate (float) - the learning rate for the optimizer.
      4. hidden_units1 (int) - the number of neurons in the first hidden layer of the custom classifier.
      5. hidden_units2 (int) - the number of neurons in the second hidden layer of the custom classifier.
      6. total_classes (int) - the number of output classes (default is 102 for flower classification).

    Returns:
      - model (nn.Module) - the newly created deep learning model with a custom classifier.
      - criterion (nn.Module) - the loss function used during training (negative log-likelihood).
      - optimizer (optim.Optimizer) - the optimizer responsible for updating the classifier's weights.
      - scheduler (optim.lr_scheduler) - the learning rate scheduler that adjusts the learning rate over time.
    """

    #Choose model base on arch
    if arch == 'vgg16':
        model = models.vgg16(weights='DEFAULT')
        input_size = model.classifier[0].in_features
    elif arch == 'vgg13':
        model = models.vgg13(weights='DEFAULT')
        input_size = model.classifier[0].in_features
    elif arch == 'densenet121':
        model = models.densenet121(weights='DEFAULT')
        input_size = model.classifier.in_features
    elif arch == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
        input_size = model.fc.in_features  # ResNet  'fc' for the classifier
        model.fc = nn.Identity()  # Remplace the present classifier
    else:
        raise ValueError(f"Architecture '{arch}' non prise en charge")

   
    for param in model.parameters():
        param.requires_grad = False
    # Replace the model's classifier with the custom classifier
    if arch in ['vgg16', 'vgg13', 'densenet121']:
        model.classifier = Classifier(input_size, hidden_units1, hidden_units2, total_classes, dropout)
    elif arch == 'resnet50':
        model.fc = Classifier(input_size, hidden_units1, hidden_units2, total_classes, dropout)
    
    # Define the loss criterion and the optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters() if arch in ['vgg16', 'vgg13', 'densenet121'] else model.fc.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    
    return model, criterion, optimizer, scheduler   
    

def check_gpu(gpu):
    '''
    Arguments: in_arg.gpu
    Returns: whether GPU is available or not 
    '''
    if not gpu:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
    print("You are using {}".format(device))
    return device


def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, epochs=5):
    """
    Train a deep learning model on a given dataset and validate its performance.
    Parameters:
      model (nn.Module) - The deep learning model to be trained.
      train_loader (DataLoader) - PyTorch DataLoader containing the training data.
      valid_loader (DataLoader) - PyTorch DataLoader containing the validation data.
      criterion (nn.Module) - The loss function used to calculate the error.
      optimizer (torch.optim.Optimizer) - The optimization algorithm used to update model weights.
      scheduler (torch.optim.lr_scheduler) - Learning rate scheduler to adjust the learning rate during training.
      device (torch.device) - The device (CPU or GPU) to perform the computations on.
      epochs (int) - The number of training epochs (default is 5).

    Returns:
      - Prints the training and validation losses, as well as validation accuracy after each epoch. The scheduler is updated at the end of each epoch.
"""
    model.to(device)
    steps = 0
    print_every=40
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            log_ps = model(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # Afficher les informations à chaque "print_every" étapes
            if steps % print_every == 0:
                print(f"Step {steps}.. "
                      f"Train loss: {running_loss/print_every:.3f}")
                running_loss = 0
 
        # Validation
        model.eval()
        val_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                log_ps = model(inputs)
                val_loss += criterion(log_ps, labels).item()
                
                # Calculer la précision
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print(f"Train loss: {running_loss/len(train_loader):.3f}.. "
              f"Validation loss: {val_loss/len(valid_loader):.3f}.. "
              f"Validation accuracy: {accuracy/len(valid_loader):.3f}")
        
        # udapte scheduler
        scheduler.step()
# Function to load a checkpoint and rebuild the model

def load_checkpoint(filepath):
    # Charger le checkpoint
    checkpoint = torch.load(filepath)
    
    # Charger le modèle pré-entraîné basé sur l'architecture spécifiée dans le checkpoint
    arch = checkpoint['arch']
    
    if arch == 'vgg16':
        model = models.vgg16(weights='DEFAULT')
        input_size = model.classifier[0].in_features
    elif arch == 'vgg13':
        model = models.vgg13(weights='DEFAULT')
        input_size = model.classifier[0].in_features
    elif arch == 'densenet121':
        model = models.densenet121(weights='DEFAULT')
        input_size = model.classifier.in_features
    elif arch == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
        input_size = model.fc.in_features
        model.fc = torch.nn.Identity()  # Remplacer le classifieur actuel
    else:
        raise ValueError(f"Architecture '{arch}' non prise en charge")
    
    # Charger le classifieur personnalisé du checkpoint
    if arch in ['vgg16', 'vgg13', 'densenet121']:
        model.classifier = checkpoint['classifier']
    elif arch == 'resnet50':
        model.fc = checkpoint['classifier']
    
    # Charger les poids du modèle
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Créer un optimiseur et charger son état
    optimizer = optim.Adam(model.classifier.parameters() if arch in ['vgg16', 'vgg13', 'densenet121'] else model.fc.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restaurer d'autres informations (comme l'indice des classes et les époques)
    class_to_idx = checkpoint['class_to_idx']
    epochs = checkpoint['epochs']
    
    # Réattribuer class_to_idx au modèle
    model.class_to_idx = class_to_idx
    
    print(f"Modèle chargé depuis {filepath}.\n"
          f"Architecture : {arch}\n"
          f"Classifieur : {model.classifier if arch in ['vgg16', 'vgg13', 'densenet121'] else model.fc}\n"
          f"Époques : {epochs}\n"
          f"Mapping des classes : {class_to_idx}")
    
    return model, optimizer, class_to_idx, epochs



# Function to process an image to the format required by the model
def process_image(image_path):
    """
    Process an image file for use in a PyTorch model.
    Parameters:
      image_path (str) - the file path to the image to be processed.
    
    Returns:
      - The processed image as a PyTorch tensor.
    """

    img_pil = Image.open(image_path)
    

    preprocess = transforms.Compose([
        transforms.Resize(256),  # Redimensionner avec le côté le plus court à 256px
        transforms.CenterCrop(224),  # Découper un carré de 224x224 au centre
        transforms.ToTensor(),  # Convertir en tensor PyTorch
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation
    ])
    
    return preprocess(img_pil)


def predict(image_path, model, device, topk=5):
    """
    Predict the class (or classes) of an image using a trained deep learning model. 
    Parameters:
      1. image_path (str) - the path to the image file to be classified.
      2. model (torch.nn.Module) - the trained deep learning model used for prediction.
      3. topk (int) - the number of top predicted classes to return (default is 5).
    
    Returns:
      - top_probs (list) - the list of the top `k` predicted class probabilities.
      - top_classes (list) - the list of the corresponding top `k` predicted class labels.
    """   
    #Predict the class (or classes) of an image using a trained deep learning model.
    
    # Mettre le modèle en mode évaluation
    model.eval()
    
    # Charger et prétraiter l'image
    img_tensor = process_image(image_path)
    
    # Ajouter une dimension batch pour correspondre aux attentes du modèle
    img_tensor = img_tensor.unsqueeze(0)
    
    # S'assurer que le modèle est sur le bon device (CPU ou GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    img_tensor = img_tensor.to(device)
    
    # Désactiver le calcul des gradients pour accélérer l'inférence
    with torch.no_grad():
        # Faire une prédiction
        output = model.forward(img_tensor)
    
    # Appliquer softmax pour obtenir les probabilités
    probabilities = torch.exp(output)
    
    # Obtenir les top k probabilités et leurs indices
    top_probs, top_indices = probabilities.topk(topk)
    
    # Déplacer les résultats sur le CPU pour traitement
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    
    # Inverser le dictionnaire class_to_idx pour obtenir les classes
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    
    # Convertir les indices en classes
    top_classes = [idx_to_class[i] for i in top_indices]
    
    return top_probs, top_classes


