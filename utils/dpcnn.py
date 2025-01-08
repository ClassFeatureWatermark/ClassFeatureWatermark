import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)

class DPCNN(BasicModule):
    """
    DPCNN for sentences classification.
    """
    def __init__(self, num_classes=4, word_num=30522):
        super(DPCNN, self).__init__()
        self.word_embedding_dimension = 100
        self.channel_size = 250
        self.embedding = nn.Embedding(word_num, self.word_embedding_dimension, padding_idx=0)
        for param in self.embedding.parameters():
            param.requires_grad = False
        self.conv_region_embedding = nn.Conv2d(1, self.channel_size,
                                               (3, self.word_embedding_dimension-10), stride=1)
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.fc = nn.Linear(self.channel_size, num_classes)

    def forward(self, x, wo_embedding=False):
        batch = x.shape[0]
        if not wo_embedding:
            x = self.embedding(x)
        x = x.unsqueeze(1)
        # print("x", x[0])

        # Region embedding
        x = self.conv_region_embedding(x)        # [batch_size, channel_size, length, 1]

        x = self.padding_conv(x)                      # pad保证等长卷积，先通过激活函数再卷积
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)

        while x.size()[-2] > 2:
            x = self._block(x)

        x = x.view(batch, self.channel_size)
        x = self.fc(x)

        return x

    def _block(self, x):
        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)

        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)

        # Short Cut
        x = x + px

        return x

    def predict(self, x):
        self.eval()
        out = self.forward(x)
        predict_labels = torch.max(out, 1)[1]
        self.train(mode=True)
        return predict_labels

