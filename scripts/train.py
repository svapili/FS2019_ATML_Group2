import torch
import numpy as np
# Define training function

def train(model, train_loader, optimizer, loss_fn, device, print_every=10, status = False):
    model.train()
    losses = []
    n_correct = 0
    n_sample = 0
    n_sample_0 = 0
    n_sample_1 = 0
    iteration = 0
    print("length of dataset :",len(train_loader.dataset))
    for iter, (images, labels) in enumerate(train_loader):
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

        losses.append(loss.item())
        n_correct += torch.sum(output.argmax(1) == labels).item()


        # DEBUG
        if status is True:
            n_sample_0 += torch.sum(output.argmax(1) == 0).item()
            n_sample_1 += torch.sum(output.argmax(1) == 1).item()
            n_sample = n_sample_0+n_sample_1
            if(n_sample>0):
                accuracy = 100.0 * n_correct / n_sample
            else:
                accuracy = np.inf

            print('Training iteration {}: loss {:.4f}, accuracy {:.4f}'.format(iteration, loss.item(), accuracy))
            print('# 0 output: {}, # 1 output: {}'.format(n_sample_0,n_sample_1))


        iteration += 1

    # plt.plot(losses)
    accuracy = 100.0 * n_correct / len(train_loader.dataset)

    return np.mean(np.array(losses)), accuracy
