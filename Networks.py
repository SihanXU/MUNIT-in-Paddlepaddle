import h5py
import numpy as np
import matplotlib.pyplot as plt
import paddle
import paddle.nn as nn 
import paddle.nn.functional as F
import paddle.vision.transforms as TF
from paddle.nn.initializer import Assign, Normal, Constant
import random
import cv2

from paddle.vision.models import vgg19
encoder = vgg19(pretrained = True)

decoder = nn.Sequential(
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(256, 256, (3, 3)),
    nn.ReLU(),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(256, 256, (3, 3)),
    nn.ReLU(),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(256, 256, (3, 3)),
    nn.ReLU(),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(128, 128, (3, 3)),
    nn.ReLU(),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(64, 64, (3, 3)),
    nn.ReLU(),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(64, 3, (3, 3)),
)

class Residual(nn.Layer):
    def __init__(self, input_output_dim, use_bias):
        super(Residual, self).__init__()
        name_scope = self.full_name()

        self.conv1 = nn.Conv2D(input_output_dim, input_output_dim, 3, bias_attr=use_bias, weight_attr=nn.initializer.Normal(mean=0, std=0.02))
        self.in1 = nn.InstanceNorm2D(input_output_dim)
        
        self.conv2 = nn.Conv2D(input_output_dim, input_output_dim, 3, bias_attr=use_bias, weight_attr=nn.initializer.Normal(mean=0, std=0.02))
        self.in2 = nn.InstanceNorm2D(input_output_dim)
    def forward(self, x_input):
        x = F.pad(x_input, [1,1,1,1], mode='reflect')
        x = self.conv1(x)
        x = self.in1(x)
        x = F.relu(x)

        x = F.pad(x, [1,1,1,1], mode='reflect')
        x = self.conv2(x)
        x = self.in2(x)
        return x + x_input

class Disc(nn.Layer):

    def __init__(self, channel=64):
        super(Disc, self).__init__()
        self.conv1 = nn.Conv2D(3,channel,4,2,1,bias_attr=True,weight_attr=nn.initializer.Normal(mean=0, std=0.02))
        self.in1 = nn.InstanceNorm2D(channel)
        self.conv2 = nn.Conv2D(channel,channel*2,4,2,1,bias_attr=True,weight_attr=nn.initializer.Normal(mean=0, std=0.02))
        self.in2 = nn.InstanceNorm2D(channel*2)
        self.conv3 = nn.Conv2D(channel*2,channel*4,4,2,1,bias_attr=True,weight_attr=nn.initializer.Normal(mean=0, std=0.02))
        self.in3 = nn.InstanceNorm2D(channel*4)
        self.conv4 = nn.Conv2D(channel*4,channel*8,4,padding=1,bias_attr=True,weight_attr=nn.initializer.Normal(mean=0, std=0.02))
        self.in4 = nn.InstanceNorm2D(channel*8)
        self.conv5 = nn.Conv2D(channel*8,1,4,padding=1,bias_attr=True,weight_attr=nn.initializer.Normal(mean=0, std=0.02))
    
    def calc_gen_loss(self, x):
        x = self.forward(x)
        return paddle.mean((x-1)**2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.in1(x)
        x = F.leaky_relu(x,negative_slope=0.2)

        x = self.conv2(x)
        x = self.in2(x)
        x = F.leaky_relu(x,negative_slope=0.2)

        x = self.conv3(x)
        x = self.in3(x)
        x = F.leaky_relu(x,negative_slope=0.2)

        x = self.conv4(x)
        x = self.in4(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        
        x = self.conv5(x)
        return x
      
   def get_mean_std(X, epsilon=1e-5):
    axes = [2,3]
    mean = paddle.mean(X, axis=axes, keepdim=True)
    standard_deviation = paddle.std(X, axis=axes, keepdim=True)
    standard_deviation = paddle.sqrt(standard_deviation + epsilon)
    return mean,standard_deviation

def adain(style, content):

    content_mean, content_std = get_mean_std(content)
    style_mean, style_std = get_mean_std(style)
    t = style_std * (content - content_mean) / content_std + style_mean
    return t
  
from paddle.vision.models import vgg19
encoder = vgg19(pretrained = True)

decoder = nn.Sequential(
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(256, 256, (3, 3)),
    nn.ReLU(),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(256, 256, (3, 3)),
    nn.ReLU(),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(256, 256, (3, 3)),
    nn.ReLU(),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(128, 128, (3, 3)),
    nn.ReLU(),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(64, 64, (3, 3)),
    nn.ReLU(),
    nn.Pad2D((1, 1, 1, 1)),
    nn.Conv2D(64, 3, (3, 3)),
)

class Residual(nn.Layer):
    def __init__(self, input_output_dim, use_bias):
        super(Residual, self).__init__()
        name_scope = self.full_name()

        self.conv1 = nn.Conv2D(input_output_dim, input_output_dim, 3, bias_attr=use_bias, weight_attr=nn.initializer.Normal(mean=0, std=0.02))
        self.in1 = nn.InstanceNorm2D(input_output_dim)
        
        self.conv2 = nn.Conv2D(input_output_dim, input_output_dim, 3, bias_attr=use_bias, weight_attr=nn.initializer.Normal(mean=0, std=0.02))
        self.in2 = nn.InstanceNorm2D(input_output_dim)
    def forward(self, x_input):
        x = F.pad(x_input, [1,1,1,1], mode='reflect')
        x = self.conv1(x)
        x = self.in1(x)
        x = F.relu(x)

        x = F.pad(x, [1,1,1,1], mode='reflect')
        x = self.conv2(x)
        x = self.in2(x)
        return x + x_input

class Disc(nn.Layer):

    def __init__(self, channel=64):
        super(Disc, self).__init__()
        self.conv1 = nn.Conv2D(3,channel,4,2,1,bias_attr=True,weight_attr=nn.initializer.Normal(mean=0, std=0.02))
        self.in1 = nn.InstanceNorm2D(channel)
        self.conv2 = nn.Conv2D(channel,channel*2,4,2,1,bias_attr=True,weight_attr=nn.initializer.Normal(mean=0, std=0.02))
        self.in2 = nn.InstanceNorm2D(channel*2)
        self.conv3 = nn.Conv2D(channel*2,channel*4,4,2,1,bias_attr=True,weight_attr=nn.initializer.Normal(mean=0, std=0.02))
        self.in3 = nn.InstanceNorm2D(channel*4)
        self.conv4 = nn.Conv2D(channel*4,channel*8,4,padding=1,bias_attr=True,weight_attr=nn.initializer.Normal(mean=0, std=0.02))
        self.in4 = nn.InstanceNorm2D(channel*8)
        self.conv5 = nn.Conv2D(channel*8,1,4,padding=1,bias_attr=True,weight_attr=nn.initializer.Normal(mean=0, std=0.02))
    
    def calc_gen_loss(self, x):
        x = self.forward(x)
        return paddle.mean((x-1)**2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.in1(x)
        x = F.leaky_relu(x,negative_slope=0.2)

        x = self.conv2(x)
        x = self.in2(x)
        x = F.leaky_relu(x,negative_slope=0.2)

        x = self.conv3(x)
        x = self.in3(x)
        x = F.leaky_relu(x,negative_slope=0.2)

        x = self.conv4(x)
        x = self.in4(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        
        x = self.conv5(x)
        return x
