import torch

'''
Test the model on the given test or validation set
Args
model:       the model we want to evaluate
test_loader: a dataloader object for the test set
loss_fn:     the loss function to use
device:      the device to use (CPU or cuda)
balance:     the threshold to assign the output to one class or the other
Return
average_loss: the average loss
accuracy:     the average accuracy
n_TP:         the number of True Positive classifications
n_TN:         the number of True Negative classifications
n_FP:         the number of False Positive classifications
n_FN:         the number of False Negative classifications
'''
def test(model, test_loader, loss_fn, device, balance=0.5):
    
    if balance is 0.5:
        print("not weighted")
    else:
        print("balance is ", balance)
          
    model.eval()
    test_loss = 0
    n_correct = 0
    n_true_output = 0
    n_false_ouput = 0
    n_TP = 0
    n_TN = 0
    n_FP = 0
    n_FN = 0
    
    balance_class1 = balance
    balance_class2 = 1-balance

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = loss_fn(output, labels)
            
            if balance is not 0.5:
                output[:,0] = (output[:,0]+100)*balance_class1
                output[:,1] = (output[:,1]+100)*balance_class2
            
            
            _, predicted = torch.max(output.data,1)

            test_loss += loss.item()
            out_argmax = output.argmax(1)

            n_correct += torch.sum(out_argmax == labels).item()
            
            n_true_output += torch.sum(out_argmax == 1).item()
            n_false_ouput += torch.sum(out_argmax == 0).item()

            n_TP += torch.sum((labels == 1) * (out_argmax == 1)).item()
            n_TN += torch.sum((labels == 0) * (out_argmax == 0)).item()
            n_FP += torch.sum((labels == 0) * (out_argmax == 1)).item()
            n_FN += torch.sum((labels == 1) * (out_argmax == 0)).item()


    average_loss = test_loss / len(test_loader)
    accuracy = 100.0 * n_correct / len(test_loader.dataset)

    return average_loss, accuracy, n_TP, n_TN, n_FP, n_FN
