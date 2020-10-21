import torch
import torch.nn.functional as F
from MultiOctConv.model import MultiOctaveConv

"""
MNist classifier with their traditional convolution replaced for M-OctConv
input:
    full: boolean that indicate if the fully conected layer should be added in to de model
"""
class M_OctConv_MNIST(torch.nn.Module):

    def __init__(self, full = True):
        self.full = full
        super(M_OctConv_MNIST, self).__init__()
        self.conv1 = MultiOctaveConv( 1, 12, 3, 
                alpha_in=0., alpha_out=0.5, beta_in=0.0,beta_out=0.0,
                conv_args = {"padding":1, "bias":False},
                downsample = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=2),
                upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.conv2 = MultiOctaveConv( 12, 12, 3, 
                    alpha_in=0.5, alpha_out=1/3, beta_in=0.0,beta_out=1/3,
                    conv_args = {"padding":1, "bias":False},
                    downsample = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=2),
                    upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
            )
        self.conv3 = MultiOctaveConv( 12, 12, 3, 
                    alpha_in=1/3, alpha_out=0.5, beta_in=1/3,beta_out=0.0,
                    conv_args = {"padding":1, "bias":False},
                    downsample = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=2),
                    upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
            )
        self.conv4 = MultiOctaveConv(12, 15, 3, 
                    alpha_in=0.5, alpha_out=0.0, beta_in=0.0,beta_out=0.0,
                    conv_args = {"padding":1, "bias":False},
                    downsample = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=2),
                    upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
            )
        if self.full:
            self.fc1 = torch.nn.Linear(735, 50)
            self.fc2 = torch.nn.Linear(50, 10)
            self.conv2_drop_h = torch.nn.Dropout2d()
            self.conv2_drop_m = torch.nn.Dropout2d()
            self.conv2_drop_l = torch.nn.Dropout2d()
            self.conv4_drop_h = torch.nn.Dropout2d()

    """
    Method that return the forward pass of only part of the layers
    inputs:
        x:  A tensor or Tuple of size 3  of tensor wich represent high, medium and low frequency features maps of M-OctConv in that order.
            If there is no channels in a level the tensor is replace by the None object.
        layer: Last layer to wich x should be pass by
    """
    def partial_forward(self,x, layer):
        x_h, x_m, x_l = self.conv1(x)
        x_h = F.relu(x_h)
        x_m = F.relu(x_m)
        if layer == 1:
            return (x_h, x_m, x_l)
        x_h, x_m, x_l =self.conv2((x_h, x_m, x_l))
        x_h = F.relu(x_h)
        x_m = F.relu(x_m)
        x_l = F.relu(x_l)
        if layer == 2:
            return (x_h, x_m, x_l)
        x_h, x_m, x_l =self.conv3((x_h, x_m, x_l))
        x_h = F.relu(F.avg_pool2d(x_h, 2))
        x_m = F.relu(F.avg_pool2d(x_m, 2))
        if layer == 3:
            return (x_h, x_m, x_l)
        x_h, x_m, x_l = self.conv4((x_h, x_m, x_l))
        x_h = F.relu(F.avg_pool2d(x_h,2))
        return (x_h)

    """
    x:  A tensor or Tuple of size 3  of tensor wich represent high, medium and low frequency features maps of M-OctConv in that order.
            If there is no channels in a level the tensor is replace by the None object.
    """
    def forward(self, x):
        x_h, x_m, x_l = self.conv1(x)
        x_h = F.relu(x_h)
        x_m = F.relu(x_m)
        x_h, x_m, x_l =self.conv2((x_h, x_m, x_l))
        if self.full:
            x_h = self.conv2_drop_h(x_h)
            x_m = self.conv2_drop_m(x_m)
            x_l = self.conv2_drop_l(x_l)
        x_h = F.relu(x_h)
        x_m = F.relu(x_m)
        x_l = F.relu(x_l)
        x_h, x_m, x_l =self.conv3((x_h, x_m, x_l))
        x_h = F.relu(F.avg_pool2d(x_h, 2))
        x_m = F.relu(F.avg_pool2d(x_m, 2))
        x_h, x_m, x_l = self.conv4((x_h, x_m, x_l))
        if self.full:
            x_h = self.conv4_drop_h(x_h)
        x_h = F.relu(F.avg_pool2d(x_h,2))
        if self.full:
            x = x_h.view(-1, 735)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
        else:
            return x_h

"""
A simple MNSIT classfier
"""
class MNIST(torch.nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.conv1 = torch.nn.Conv2d( 1,  12, kernel_size=3, padding=1, bias= False)
        self.conv2 = torch.nn.Conv2d(12,  12, kernel_size=3, padding=1, bias= False)
        self.conv3 = torch.nn.Conv2d(12,  12, kernel_size=3, padding=1, bias= False)
        self.conv4 = torch.nn.Conv2d(12, 15, kernel_size=3, padding=1, bias= False)
        self.conv2_drop = torch.nn.Dropout2d()
        self.conv4_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(735, 50)
        self.fc2 = torch.nn.Linear(50, 10)
        
    """
    Function that make a forward pass only in the fully conected layer of the model
    x: A tensor with shape [-1,-735]
    """
    def fc_layers(self,x):
        x = x.view(-1, 735)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)

    """
    Method that return the forward pass of only part of the layers
    inputs:
        x: A tensor that represent a MNIST image
        layer: Last layer to wich x should be pass by
    """
    def partial_forward(self,x, layer):
        x = F.relu(self.conv1(x))
        if layer == 1:
            return x
        x = F.relu(self.conv2(x))
        if layer == 2:
            return x
        x = F.relu(F.avg_pool2d(self.conv3(x), 2))
        if layer == 3:
            return x
        x = F.relu(F.avg_pool2d(self.conv4(x), 2))
        return x

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2_drop(self.conv2(x)))
        x = F.relu(F.avg_pool2d(self.conv3(x), 2))
        x = F.relu(F.avg_pool2d(self.conv4_drop(self.conv4(x)), 2))
        x = x.view(-1, 735)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)