import torch
import torch.nn as nn
import torchvision
import phyre

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')

class ActionNetwork(nn.Module):

    def __init__(self, action_size, output_size, hidden_size):
        super().__init__()
        self.layer  = nn.Linear(action_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size) 

    def forward(self, tensor):
        tensor = nn.functional.relu(self.layer(tensor), inplace=True)
        output = self.output(tensor)
        return output


class FilmActionNetwork(nn.Module):

    def __init__(self, action_size, output_size, hidden_size):
        super().__init__()
        # Output size should be doulbed to extract the beta & gamma
        output_size = output_size * 2
        self.net = ActionNetwork(action_size, output_size, hidden_size)

    def forward(self, actions, image):
        beta, gamma = torch.chunk(self.net(actions).unsqueeze(-1).unsqueeze(-1), chunks=2, dim=1)
        return image * beta + gamma


class ResNet18FilmAction(nn.Module):

    def __init__(self,
                 action_size,
                 action_hidden_size):
        super().__init__()
        # output_size is designated as a number of channel in resnet
        output_size = 256
        net = torchvision.models.resnet18(pretrained=False)
        conv1 = nn.Conv2d(phyre.NUM_COLORS,64, kernel_size=7, stride=2, padding=3, bias=False)
        self.register_buffer('embed_weights', torch.eye(phyre.NUM_COLORS))
        self.stem = nn.Sequential(conv1, net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        self._action_network = FilmActionNetwork(action_size = action_size,
                                                 output_size = output_size,
                                                 hidden_size = action_hidden_size)
        self.action_networks = [None, None, None, self._action_network]
        # number of channel in the last resnet is 512
        self.reason = nn.Linear(512, 1)

    @property
    def device(self):
        if hasattr(self, 'parameters') and next(self.parameters()).is_cuda:
            return 'cuda'
        else:
            return 'cpu'

    def preprocess(self, observations):
        image = self._image_colors_to_onehot(observations)
        features = self.stem(image)
        # Forward
        for stage, act_layer in zip(self.stages, self.action_networks):
            if act_layer is not None:
                break
            features = stage(features)
        else:
            features = nn.functional.adaptive_max_pool2d(features, 1)
        return features

    def forward(self, observations, actions, preprocessed = None):
        if preprocessed is None:
            features= self.preprocess(observations)
        else:
            features = preprocessed
        actions = actions.to(features.device)
        # Fusion the image features & action features with Film Network
        skip = True
        for stage, film_layer in zip(self.stages, self.action_networks):
            if film_layer is not None:
                skip = False
                features = film_layer(actions, features)
            if skip:
                continue
            # perform conv4 for the last layer
            features = stage(features)
        features = nn.functional.adaptive_max_pool2d(features, 1)
        features = features.flatten(1)
        
        # If multiple actions are provided with a given image, shape should be adjusted
        if features.shape[0] == 1 and actions.shape[0] != 1:
            features = features.expand(actions.shape[0], -1)
        return self.reason(features).squeeze(-1)

    def ce_loss(self, decisions, targets):
        targets = targets.to(dtype=torch.float, device=decisions.device)
        return nn.functional.binary_cross_entropy_with_logits(decisions, targets)

    def _image_colors_to_onehot(self, indices):
        onehot = torch.nn.functional.embedding(
            indices.to(dtype=torch.long, device=self.embed_weights.device),
            self.embed_weights)
        onehot = onehot.permute(0, 3, 1, 2).contiguous()
        return onehot


class ResNet18PhysicalQA(nn.Module):
    
    def __init__(self):
        pass