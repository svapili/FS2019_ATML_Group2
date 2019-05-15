import torch
#from sklearn.metrics import f1_score, balanced_accuracy_score

# Define testing function
def test(model, test_loader, loss_fn, device):
    '''
    Tests the model on data from test_loader
    '''
    model.eval()
    test_loss = 0
    n_correct = 0
    n_true_output = 0
    n_false_ouput = 0
    n_TP = 0
    n_TN = 0
    n_FP = 0
    n_FN = 0



    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = loss_fn(output, labels)

            _, predicted = torch.max(output.data,1)

            test_loss += loss.item()
            out_argmax = output.argmax(1)

            n_correct += torch.sum(out_argmax == labels).item()
            #n_correct += np.sum(output.argmax(1).numpy()==labels.numpy())

            n_true_output += torch.sum(out_argmax == 1).item()
            n_false_ouput += torch.sum(out_argmax == 0).item()

            n_TP += torch.sum((labels == 1) * (out_argmax == 1)).item()
            n_TN += torch.sum((labels == 0) * (out_argmax == 0)).item()
            n_FP += torch.sum((labels == 0) * (out_argmax == 1)).item()
            n_FN += torch.sum((labels == 1) * (out_argmax == 0)).item()



    average_loss = test_loss / len(test_loader)
    accuracy = 100.0 * n_correct / len(test_loader.dataset)

#   print('Test average loss: {:.4f}, accuracy: {:.3f}'.format(average_loss, accuracy))
    return average_loss, accuracy, n_TP, n_TN, n_FP, n_FN