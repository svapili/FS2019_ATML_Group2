import torch
# Define training function

def train(model, train_loader, optimizer, loss_fn, device, print_every=100):
    model.train()
    losses = []
    n_correct = 0
    for iteration, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        '''
        print(train_loader.batch_size)
        print(labels.size())
        print(output.shape)
        '''
        optimizer.zero_grad()
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        # DEBUG
        if iteration % print_every == 0:
            print('Training iteration {}: loss {:.4f}'.format(iteration, loss.item()))

        losses.append(loss.item())
        n_correct += torch.sum(output.argmax(1) == labels).item()

    # plt.plot(losses)
    accuracy = 100.0 * n_correct / len(train_loader.dataset)

    return np.mean(np.array(losses)), accuracy