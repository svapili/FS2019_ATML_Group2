import torch
import torch.nn as nn
import torch.optim as optim

'''
Set the correct "requires_grad" parameters to enable finetunning
model:      the model we want to train
finetune:   a boolean telling if finetuning is active
'''
def set_parameter_requires_grad(model, finetune):
    if not finetune:
        for param in model.parameters():
            param.requires_grad = False


'''
Create a new optimizer object for the given parameters. Finetuning may be
enabled for pretrained models.
model:          the model for which to setup the optimizer
learning_rate:  the learning rate
optimizer_name: a string containing the name of the optimizer 
pretrained:     use a pretrained model if set to True
finetune:       activate finetunning if set to True (only for pretrained models)
Return
optimizer: the new optimizer object
'''
def setup_optimizer(model, learning_rate=0.001, optimizer_name="Adam", pretrained=False, finetune=False): 
    if pretrained:
        num_classes=2
        if model.__class__.__name__ == "ResNet":
            num_ftrs = model.fc.in_features  
            set_parameter_requires_grad(model, finetune)
            model.fc = nn.Linear(num_ftrs, num_classes)
            params_to_update = model.parameters()
        elif model.__class__.__name__ =="AlexNet":
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
