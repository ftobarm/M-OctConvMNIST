import torch
import torch.nn.functional as F
from sys import stderr
log_interval = 10

"""
Function that set the weights of convolution in a M_OctConv_MNIST Model
inputs:
    model: M_OctConv_MNIST to be modified
    weights: Values of the weights to be set in the convolution of the model
    layer : Number of the convolution layer that will be set
    conv_type: Type of convolution in the layer that will be modified 
"""
def set_conv(model, weights, layer, conv_type):
    if layer == 1:
        model = model.conv1
    elif layer == 2:
        model = model.conv2
    elif layer == 3:
        model = model.conv3
    elif layer == 4:
        model = model.conv4
    if conv_type == "H-H":
        model = model.conv_h2h
    elif conv_type == "M-H":
        model = model.conv_l2h
    elif conv_type == "H-M":
        model = model.conv_h2l
    elif conv_type == "M-M":
        model = model.conv_l2l
    elif conv_type == "L-M":
        model = model.conv_ll2l
    elif conv_type == "M-L":
        model = model.conv_l2ll
    elif conv_type == "L-L":
        model = model.conv_ll2ll
    with torch.no_grad():
        model.weight = weights


"""
Function that train a  whole MNIST classfier for one epoch
inputs:
    epochs: Number of the epcoh that is trained
    network: Model to be trained
    optimizer: Optimizer that help the training
    train_losess: List that save the loss values
    train_counter: list that save the number of image trained 
    train_loader: data loader for the training phase
    cuda: flag to use cuda device
    log_interval: Number that control into how many estep the data is loged
"""
def train(epoch, network, optimizer, train_losses, train_counter, train_loader, cuda=False, log_interval=10):
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), file=stderr)
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
"""
Function that test a trained  MNIST classfier
inputs:
    network: The model wich their conv layer will be tested
    test_losess: List that save the loss values 
    test_loader: data loader for the test phase
    test_accuracy: List that save the accuracy of the test
    save: Flag to save or not the results of the testing
    cuda: flag to use cuda device
"""
def test(network, test_losses, test_loader, test_accuracy, save=True, cuda=False):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for _, (data, target) in enumerate(test_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            if cuda:
                data, target = data.cpu(), target.cpu()
    test_loss /= len(test_loader.dataset)
    if save:
        test_losses.append(test_loss)
        test_accuracy.append((correct.cpu().item(),len(test_loader.dataset)))
        
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)
"""
Function that train only the convolution layers of a  MNIST classfier for one epoch and use the fully connected
layers of a secondary network
inputs:
    epochs: Number of the epcoh that is trained
    network: The model wich their conv layer will be trained
    secondary network: Trained MNIST model that provide the fully connected layer for the training 
    optimizer: Optimizer that help the training
    train_losess: List that save the loss values
    train_counter: list that save the number of image trained 
    train_loader: data loader for the training phase
    cuda: flag to use cuda device
    log_interval: Number that control into how many estep the data is loged
"""
def partial_train(epoch, network, second_network, optimizer, train_losses, train_counter, train_loader, cuda=False, log_interval=10):
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data = data.cuda()
        with torch.no_grad():
            target = second_network.partial_forward(data, 4)
        optimizer.zero_grad()
        output = network(data)
        loss = torch.mean(torch.abs(output - target))
        loss.backward()
        optimizer.step()
        if cuda:
            data = data.cpu()
            target= target.cpu()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), file=stderr)
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

"""
Function that test the trained  convolution layers of a MNIST classfier using the fully connected
layers of a secondary network
inputs:
    network: The model wich their conv layer will be tested
    secondary network: Trained MNIST model that provide the fully connected layer for the testing 
    test_losess: List that save the loss values 
    test_loader: data loader for the test phase
    test_accuracy: List that save the accuracy of the test
    batch_size_test: batch size for the testing
    save: Flag to save or not the results of the testing
    cuda: flag to use cuda device
"""
def partial_test(network, second_network, test_losses, test_loader, test_accuracy, batch_size_test, save=True, cuda=False):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for _, (data, target) in enumerate(test_loader):
            if cuda:
                data, target = data.cuda(), target.cuda(),
            custom_target = second_network.partial_forward(data ,4)
            output = network(data)
            test_loss += torch.abs(output - custom_target)
            output = second_network.fc_layers(output)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            if cuda:
                data, target, custom_target = data.cpu(), target.cpu(), custom_target.cpu()
    test_loss = (batch_size_test/len(test_loader.dataset))* test_loss.mean()
    if save:
        test_losses.append(test_loss.item())
        test_accuracy.append((correct.cpu().item(),len(test_loader.dataset)))

    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return  100. * correct / len(test_loader.dataset)

"""
Function that train only one convolution of a M-OctConv-MNIST classfier using a trained MNIST Classifer to get the 
expected outputs
inputs:
    epochs: Number of the epcoh that is trained
    network: The conv layer that will be trained
    conv_type : kind of convluition inside of a OctConv to be trained
    vanilla_net: Trained MNIST model that provide the goal outputs for each layer
    oct_net: The model that produce the input for the conv layer form a  mnist dataset value
    optimizer: Optimizer that help the training
    train_losess: List that save the loss values
    train_counter: list that save the number of image trained 
    train_loader: data loader for the training phase
    dict_networks: dictionary with trained conv from M-OctConv Model that is being trained
    up_sample2 : Function that applied the upsample
    up_sample4 : Function that applied the double upsample 
    cuda: flag to use cuda device
    log_interval: Number that control into how many estep the data is loged
"""
def custom_train(epoch, network, layer, conv_type, 
                 vanilla_net, oct_net, optimizer, 
                 train_losses, train_counter, train_loader, dict_networks, up_sample2, up_sample4,
                  cuda=False, log_interval=10):
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data = data.cuda()
        target = vanilla_net.partial_forward(data, layer)
        with torch.no_grad():
            x_h, x_m, x_l = oct_net.partial_forward(data, layer-1)
        optimizer.zero_grad()
        ##Select the channels of the targets that are aproximated for  each frequency 
        if layer == 2:
            if conv_type[-1] == "H":
                target = target[:,:4,:,: ]
            elif conv_type[-1] == "M":
                target = target[:,4:8,:,:]
            elif conv_type[-1] == "L":
                target = target[:,8:,:,:]
        elif layer == 3:
            if conv_type[-1] == "H":
                target = target[:,:6,:,:]
            elif conv_type[-1] == "M":
                target = target[:,6:,:,:]
        
        #Modified in order that the highers frequency levels have to reduce the errors of 
        #the lower frequency levels
        if conv_type == "H-H" and "M-H" in  dict_networks[layer]:
            with torch.no_grad():
                if layer == 3 or layer == 4:
                    target -= dict_networks[layer]["M-H"](x_m)
                else:
                    target -= up_sample2(dict_networks[layer]["M-H"](x_m))
        elif conv_type == "H-M" and "M-M" in  dict_networks[layer]:
            with torch.no_grad():
                if layer == 3 or layer == 4:
                    target -= dict_networks[layer]["M-M"](x_m)
                else:
                    target -= up_sample2(dict_networks[layer]["M-M"](x_m))
            
        elif conv_type == "M-M" and "L-M" in  dict_networks[layer]:
            with torch.no_grad():
                if layer == 3 or layer == 4:
                    target -= up_sample2(dict_networks[layer]["L-M"](x_l))
                else:
                    target -= up_sample4(dict_networks[layer]["L-M"](x_l))
        elif conv_type == "M-L" and "L-L" in  dict_networks[layer]:
            with torch.no_grad():
                if layer == 3 or layer == 4:
                    target -= up_sample2(dict_networks[layer]["L-L"](x_l))
                else:
                    target -= up_sample4(dict_networks[layer]["L-L"](x_l))

        ##applied the MaxPool/upsample layers of the original model if is necesary
        if conv_type[0] == "H":
            output = network(x_h)
        elif conv_type[0] == "M":
            output = up_sample2(network(x_m))
        elif conv_type[0] == "L":
            output = up_sample4(network(x_l))
        if layer == 3 or layer == 4:
            output = F.max_pool2d(output,2)

        loss = torch.mean(torch.abs(output - target))
        loss.backward()
        optimizer.step()

        if cuda:
            data = data.cpu()
            target= target.cpu()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), file=stderr)
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
