import torch

# Define testing function
def test(model, test_loader, loss_fn, device):
    '''
    Tests the model on data from test_loader
    '''
    model.eval()
    test_loss = 0
    n_correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = loss_fn(output, labels)

            _, predicted = torch.max(output.data,1)

            test_loss += loss.item()
            n_correct += torch.sum(output.argmax(1) == labels).item()
            #n_correct += np.sum(output.argmax(1).numpy()==labels.numpy())

    average_loss = test_loss / len(test_loader)
    accuracy = 100.0 * n_correct / len(test_loader.dataset)
#   print('Test average loss: {:.4f}, accuracy: {:.3f}'.format(average_loss, accuracy))
    return average_loss, accuracy