import torch
import torch.nn as nn
import torch.optim as optim

def set_parameter_requires_grad(model, finetune):
    if not finetune:
        for param in model.parameters():
            param.requires_grad = False

def setup_optimizer(model, learning_rate=0.001, optimizer_name="Adam", pretrained=False, finetune=False): 
    if pretrained:
        num_classes=2
        if model.__class__.__name__ == "ResNet":
            #ft_model = model
            num_ftrs = ft_model.fc.in_features  
            set_parameter_requires_grad(model, finetune)
            model.fc = nn.Linear(num_ftrs, num_classes)
            params_to_update = model.parameters()
        elif model.__class__.__name__ =="AlexNet":
            #ft_model = model
            num_ftrs = model.classifier[6].in_features
            set_parameter_requires_grad(model, finetune)
            model.classifier[6] = nn.Linear(num_ftrs,num_classes)
            params_to_update = model.parameters()
        else:
            print("Invalid model name...")

        if optimizer_name=="Adam":
            optimizer = torch.optim.Adam(params_to_update, lr=learning_rate)
        elif optimizer_name=="SGD":
            optimizer = torch.optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)
        else:
            print("Invalid optimizer name")

    else:
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    return optimizer