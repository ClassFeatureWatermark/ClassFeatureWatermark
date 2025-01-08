"""resnet in pytorch

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""
# Based on https://github.com/weiaicunzai/pytorch-cifar100

import torch
import torch.nn as nn
import torch.nn.functional as F
def pairwise_euclid_distance(A):
    sqr_norm_A = A.pow(2).sum(1).unsqueeze(0)
    sqr_norm_B = A.pow(2).sum(1).unsqueeze(1)
    inner_prod = torch.matmul(A, A.t())
    tile_1 = sqr_norm_A.repeat([A.shape[0], 1])
    tile_2 = sqr_norm_B.repeat([1, A.shape[0]])
    return tile_1 + tile_2 - 2 * inner_prod
    # return torch.cdist(A, A, 2)


def cosine_distance_torch(x1, eps=1e-8):
    x2 = x1
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
def calculate_snnl_torch(x, y, t, metric='euclidean'):
    x = F.relu(x)
    same_label_mask = y.eq(y.unsqueeze(1)).squeeze()
    if metric == 'euclidean':
        dist = pairwise_euclid_distance(x.contiguous().view([x.shape[0], -1]))
    elif metric == 'cosine':
        dist = cosine_distance_torch(x.contiguous().view([x.shape[0], -1]))
    else:
        raise NotImplementedError()
    exp = torch.clamp((-(dist / t)).exp() - torch.eye(x.shape[0]).cuda(), 0, 1)
    prob = (exp / (0.00001 + exp.sum(1).unsqueeze(1))) * same_label_mask
    loss = - (0.00001 + prob.mean(1)).log().mean()
    return loss

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34"""

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),#inplace=True
            nn.Conv2d(
                out_channels,
                out_channels * BasicBlock.expansion,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=False)(self.residual_function(x) + self.shortcut(x))#inplace=True


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers"""

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),#inplace=True
            nn.Conv2d(
                out_channels,
                out_channels,
                stride=stride,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),#inplace=True
            nn.Conv2d(
                out_channels,
                out_channels * BottleNeck.expansion,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * BottleNeck.expansion,
                    stride=stride,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))#inplace=True


class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),#nn.ReLU(inplace=True),
        )
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, feature=False, return_activation=False):
        output = self.conv1(x)
        activation1 = output
        output = self.conv2_x(output)
        activation2 = output
        output = self.conv3_x(output)
        activation3 = output
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        # output = output.view(output.size(0), -1)
        output = torch.flatten(output, 1)
        if feature:
            hidden = output
        output = self.fc(output)

        if return_activation:
            return output, activation1, activation2, activation3
        elif feature:
            return output, hidden
        else:
            return output

    # def reps(self, x):
    #     outputs = []
    #     output = self.conv1(x)
    #     output = self.conv2_x(output)
    #     output = self.conv3_x(output)
    #     output = self.conv4_x(output)
    #     output = self.conv5_x[0](output)
    #     for layer in [self.conv5_x]:
    #         for block in layer:
    #             identity = x
    #             out = block.conv1(x)
    #             out = block.bn1(out)
    #             out = block.relu(out)
    #             outputs.append(out.clone())
    #             out = block.conv2(out)
    #             out = block.bn2(out)
    #             x = out + identity  # shortcut 加法
    #             x = block.relu(x)
    #     output = self.avg_pool(output)
    #     output = torch.flatten(output, 1)
    #     output = self.fc(output)
    #     return output

    def feature(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        # output = output.view(output.size(0), -1)
        output = torch.flatten(output, 1)
        # output = self.fc(output)
        return output

    def snnl_trigger(self, x, w, temperateure_tensor=None): #[1, 1, 1]
        output = self.forward(x, return_activation=True)#ewe=True
        activations = [output[i+1] for i in range(len(output)-1)]
        inv_temperatures = [100.0 / i for i in temperateure_tensor]
        losses = [calculate_snnl_torch(activations[i], w, inv_temperatures[i]).requires_grad_() for i in
                  range(len(activations))]
        return losses

def resnet18(num_classes=10):
    """return a ResNet 18 object"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes=10):
    """return a ResNet 34 object"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes=10):
    """return a ResNet 50 object"""
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes=10):
    """return a ResNet 101 object"""
    return ResNet(BottleNeck, [3, 4, 23, 3])


def resnet152():
    """return a ResNet 152 object"""
    return ResNet(BottleNeck, [3, 8, 36, 3])
