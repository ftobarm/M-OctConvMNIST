import torch
import torchvision
import argparse
from Models_mnist import MNIST, M_OctConv_MNIST
from Mnist_test_func import set_conv, train, custom_train, partial_train, partial_test, test
import pickle
import os




def main(config):
    results = [0.0, 0.0, 0.0, 0.0]
    #create the data loader
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
        batch_size=config.batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files/', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
        batch_size=config.batch_size_test, shuffle=True)

    #Initiate the Models
    nets =[
    
            (MNIST(), "vanilla"), #traditional model 
           (M_OctConv_MNIST(), "M-OctConv-full"), #M-OctMNIST-v1 (the whole model)
           (M_OctConv_MNIST(full=False), "M-OctConv-OnlyConv") #M-OctMNSIT (Only the M-OctConv layer)
        
          ]
    #train for ech model
    results_counter = 0
    for network, name in nets:
        if config.cuda:
            network = network.cuda()
        #create the directories
        save_path = "./results/"+name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #create the optimizer for the model
        optimizer = torch.optim.SGD(network.parameters(), lr=config.learning_rate,
                              momentum=config.momentum)

        train_losses = []
        train_counter = []
        test_losses = []
        test_counter = [i*len(train_loader.dataset) for i in range(config.n_epochs + 1)]
        test_accuracy = []
        print(name)
        #train the only OctConv model
        if name == "M-OctConv-OnlyConv":
            #load a trained model of MNIST-classifer
            second_network = MNIST()
            second_network.load_state_dict(torch.load("./results/vanilla/model.pth"))
            second_network.eval()
            if config.cuda:
                second_network = second_network.cuda()
            #print the results of the no trained network, just for control
            results[results_counter] = partial_test(network, second_network, test_losses,test_loader, test_accuracy, config.batch_size_test, cuda=config.cuda)
            
            for epoch in range(1, config.n_epochs + 1):
                partial_train(epoch, network, second_network, optimizer, train_losses, train_counter, train_loader, cuda=config.cuda)
                results[results_counter] = partial_test(network, second_network, test_losses,test_loader, test_accuracy, config.batch_size_test, cuda=config.cuda)
            if config.cuda:
                second_network =  second_network.cpu()
        #train the full models
        else:
            results[results_counter] = test(network,test_losses,test_loader,test_accuracy, cuda=config.cuda)
            for epoch in range(1, config.n_epochs + 1):
                train(epoch, network, optimizer, train_losses, train_counter, train_loader, cuda=config.cuda)
                results[results_counter] = test(network,test_losses, test_loader,test_accuracy, cuda=config.cuda)
        if config.cuda:
            network = network.cpu()   
        if config.save:
            with open('{}/train_losses.pickle'.format(save_path), 'wb') as f: 
                pickle.dump(train_losses, f)
            with open('{}/train_counter.pickle'.format(save_path), 'wb') as f: 
                pickle.dump(train_counter, f)
            with open('{}/test_losses.pickle'.format(save_path), 'wb') as f: 
                pickle.dump(test_losses, f)
            with open('{}/test_counter.pickle'.format(save_path), 'wb') as f: 
                pickle.dump(test_counter, f)
            with open('{}/test_accuracy.pickle'.format(save_path), 'wb') as f: 
                pickle.dump(test_accuracy, f)
            torch.save(network.state_dict(), '{}/model.pth'.format(save_path)) 
        results_counter +=1  

    #train M-OctMNIST-v3(all the convolution are trianed in a special order)
    test_model = M_OctConv_MNIST(full=False)
    #load the vanilla model
    second_network = MNIST()
    second_network.load_state_dict(torch.load("./results/vanilla/model.pth"))
    second_network.eval()
    if config.cuda:
        test_model = test_model.cuda()
        second_network =second_network.cuda()
    #dict that contained the trained Octconv layer
    dict_networks = dict()
    #dicts that contains the conv inside each OctConv layer
    dict_networks[2] = dict()
    dict_networks[3] = dict()
    dict_networks[4] = dict()
    #set the first layer of the model equal to the vanilla model
    set_conv(test_model, torch.nn.Parameter(second_network.conv1.weight[:6]), 1, "H-H")
    set_conv(test_model, torch.nn.Parameter(second_network.conv1.weight[6:]), 1, "H-M")
    """
    List of tuples that contain the data for each layer with form (#Layer, conv-info)
    where conv-info: ((channels_in_1, channels_out_1, type_conv_1), ..., (channels_in_n, channels_out_n, type_conv_n))
    """
    layers =[(2,((6, 4,  "M-L"), (6, 4, "M-M") , (6, 4, "M-H"), (6, 4, "H-M"),  (6, 4, "H-H"))),  
             (3,((4, 6,  "L-M"), (4, 6, "M-M") , (4, 6, "M-H"),  (4, 6, "H-M"),  (4, 6, "H-H"))),  
             (4,((6, 15, "M-H"), (6, 15, "H-H")))
            ]
    #flag to manage the control test before the training
    flag =True
    for layer, conv_list in layers:
        for c_in, c_out, conv_type in conv_list: 
            save_path = "./results/{}-{}".format(layer,conv_type)
            if not os.path.exists(save_path):
                os.makedirs(save_path) 
            train_losses = []
            train_counter = []
            test_losses = []
            test_counter = [i*len(train_loader.dataset) for i in range(config.n_epochs + 1)]
            test_accuracy = []
            network = torch.nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
            if config.cuda:
                network= network.cuda()
            optimizer = torch.optim.SGD(network.parameters(), lr=config.learning_rate,
                                  momentum=config.momentum)
            up_sample2 = torch.nn.Upsample(scale_factor=2, mode='nearest')
            up_sample4 = torch.nn.Upsample(scale_factor=4, mode='nearest')
            
            if flag:
                results[results_counter] = partial_test(test_model, second_network, test_losses,test_loader, test_accuracy, config.batch_size_test, cuda=config.cuda)
                flag = False
            for epoch in range(1, config.n_epochs + 1):
                custom_train(epoch, network, layer, conv_type,
                             second_network, test_model, optimizer,
                             train_losses, train_counter, train_loader, dict_networks, up_sample2, up_sample4, cuda=config.cuda)
                dict_networks[layer][conv_type] = network
                set_conv(test_model, network.weight, layer, conv_type)
                results[results_counter] = partial_test(test_model, second_network, test_losses,test_loader, test_accuracy, config.batch_size_test, cuda=config.cuda)
            if config.save:
                with open('{}/train_losses.pickle'.format(save_path), 'wb') as f: 
                    pickle.dump(train_losses, f)
                with open('{}/train_counter.pickle'.format(save_path), 'wb') as f: 
                    pickle.dump(train_counter, f)
                with open('{}/test_losses.pickle'.format(save_path), 'wb') as f: 
                    pickle.dump(test_losses, f)
                with open('{}/test_counter.pickle'.format(save_path), 'wb') as f: 
                    pickle.dump(test_counter, f)
                with open('{}/test_accuracy.pickle'.format(save_path), 'wb') as f: 
                    pickle.dump(test_accuracy, f)
                torch.save(network.state_dict(), '{}/model.pth'.format(save_path))
    print("MNIST Accuracy: {:.0f}%\nM-OctMNIST-v1 Accuracy: {:.0f}%\nM-OctMNIST-v2 Accuracy: {:.0f}%\nM-OctMNIST-v3 Accuracy: {:.0f}%\n ".format(results[0], results[1], results[2], results[3]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epcho to train')
    parser.add_argument('--batch_size_train', type=int, default=64, help='batch size for the Traning')
    parser.add_argument('--batch_size_test', type=int, default=1000, help='batch size for the Testing')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learing rate for the training')
    parser.add_argument('--momentum', type=float, default=0.5, help='momentum for the optimizer')
    parser.add_argument('--random_seed', type=int, default=1, help="random seed for the taining shuffle")
    parser.add_argument('--save', type=bool, default=True, help='flag to save the results')
    parser.add_argument('--cuda', type=bool, default=False, help='indicate if use cuda device')
    
    config = parser.parse_args()
    main(config)