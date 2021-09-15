from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision import models

class Vgg19Activations(torch.nn.Module):
    "Vgg19 with exposed layer activations"

    def __init__(self, model_file, layers_to_use, num_classes):
        super().__init__()

        if not layers_to_use == ["iterate"]:
            self.layers_to_use = layers_to_use
        else:
            self.layers_to_use = ["all"]

        vgg19 = models.vgg19(pretrained=False)
        vgg19.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1) # input layer one channel
        vgg19.classifier._modules['6'] = nn.Linear(in_features=4096, out_features=num_classes, bias=True) # output layer to correct classes
        vgg19.load_state_dict(torch.load(model_file))

        # layer names
        feature_layers = ["conv1_1", "relu1_1",  "conv1_2", "relu1_2", "mp1",
                "conv2_1", "relu2_1",  "conv2_2", "relu2_2", "mp2",
                "conv3_1", "relu3_1",  "conv3_2", "relu3_2", "conv3_3", "relu3_3",  "conv3_4", "relu3_4", "mp3",
                "conv4_1", "relu4_1",  "conv4_2", "relu4_2", "conv4_3", "relu4_3",  "conv4_4", "relu4_4", "mp4",
                "conv5_1", "relu5_1",  "conv5_2", "relu5_2", "conv5_3", "relu5_3",  "conv5_4", "relu5_4", "mp5"]

        classifier_layers = ["linear1", "reluC_1", "dropout1", "linear2", "reluC_2", "dropout2", "output"]

        # add layers to ordered dict
        self.layer_dict = torch.nn.ModuleDict()

        for i, layer in enumerate(feature_layers):
            self.layer_dict[layer] = vgg19.features[i]

        self.layer_dict["avgpool"] = vgg19.avgpool 

        for i, layer in enumerate(classifier_layers):
            self.layer_dict[layer] = vgg19.classifier[i]

    def forward(self, x):

        layer_input = x
        output_dict = {}

        # iterate over layers in ordered dict
        for layer_name, layer in self.layer_dict.items():

            # get output for this layer
            layer_output = layer(layer_input)

            if layer_name in self.layers_to_use or self.layers_to_use == ["all"]:
                # add dict item to output
                output_dict[layer_name] = layer_output

            if not layer_name == "avgpool":
                # output to input of next layer
                layer_input = layer_output
            else:
                # as above but flattened
                layer_input = torch.flatten(layer_output, 1)

        return output_dict
