import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):

    def __init__(self, window=5, dim=6, lstm_units=64, num_layers=1):
        super(CNNLSTMModel, self).__init__()
        self.conv1d = nn.Conv1d(dim, lstm_units, 1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.05)
        self.lstm = nn.LSTM(lstm_units, lstm_units, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(lstm_units * 2, 1)
        self.act4 = nn.Tanh()
    def forward(self, x):
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        x = self.act1(x)
        x = self.maxPool(x)  # bs, lstm_units, 1
        x = self.drop(x)
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        x = self.act2(x)
        x = x.squeeze(dim=1)  # bs, 2*lstm_units
        x = self.cls(x)
        x = self.act4(x)
        return x


class CNNLSTMModel_ECA(nn.Module):

    def __init__(self, window=2, dim=13, lstm_units=128, num_layers=1):
        super(CNNLSTMModel_ECA, self).__init__()
        self.conv1d = nn.Conv1d(dim, lstm_units, 1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(lstm_units, lstm_units, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.act2 = nn.Tanh()
        self.attn = nn.Linear(lstm_units * 2, lstm_units * 2)
        self.act3 = nn.Sigmoid()
        self.cls = nn.Linear(lstm_units * 2, 1)
        self.act4 = nn.Tanh()

    def forward(self, x):
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        x = self.act1(x)
        x = self.maxPool(x)  # bs, lstm_units, 1
        x = self.drop(x)
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        x = self.act2(x)
        x = x.squeeze(dim=1)  # bs, 2*lstm_units
        attn = self.attn(x)  # bs, 2*lstm_units
        attn = self.act3(attn)
        x = x * attn
        x = self.cls(x)
        x = self.act4(x)
        return x


class CNNLSTMModel_SE(nn.Module):
    def __init__(self, window=5, dim=13, lstm_units=256, num_layers=1):
        super(CNNLSTMModel_SE, self).__init__()
        self.conv1d = nn.Conv1d(dim, lstm_units, 1)  # 一层卷积
        self.act1 = nn.Sigmoid()   # 非线性变换 Non-linear transformation
        # 定义一个一维最大池化层，池化窗口大小等于时间步数 window
        # Define a one-dimensional max pooling layer
        # with a kernel size equal to the number of time steps window.
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        # 定义一个 Dropout 层，用于防止过拟合。p=0.01 表示随机丢弃1%的神经元输出
        # Define a Dropout layer to prevent overfitting. p=0.01
        # indicates that 1% of the neuron outputs are randomly dropped.
        self.drop = nn.Dropout(p=0.05)
        self.lstm = nn.LSTM(lstm_units, lstm_units, batch_first=True, num_layers=num_layers, bidirectional=True)
        # 定义一个 Tanh 激活函数，用于 LSTM 层之后的非线性变换
        # Define a Tanh activation function for non-linear
        # transformation after the LSTM layer.
        self.act2 = nn.Tanh()
        # 定义一个线性层，用于将 LSTM 层的输出转换为最终的预测结果。
        # 由于 LSTM 是双向的，所以它的输出单元数是 lstm_units * 2

        # Define a linear layer to convert the output of the LSTM layer
        # into the final prediction result. Since the LSTM is bidirectional,
        # its output unit count is lstm_units * 2.

        # self.attn = nn.Linear(lstm_units * 2, lstm_units * 2)
        # self.act3 = nn.Sigmoid()

        self.cls = nn.Linear(lstm_units * 2 * num_layers, 1)
        self.act4 = nn.Tanh()  # 非线性变换
        # 定义一个线性层，用于计算注意力权重。它将时间步的平均值映射回相同维度的空间
        # Define a linear layer to compute attention weights.
        # It maps the mean of the time steps back to the same dimensional space.
        self.se_fc = nn.Linear(window, window)

    def forward(self, x):
        # 调整输入数据的维度，使其适应 PyTorch 的维度要求。-1 和 -2 分别表示最后一个和倒数第二个维度，这里将特征维度和时间步维度交换
        # Adjust the dimensions of the input data to meet the dimension requirements of PyTorch.
        # -1 and -2 represent the last and second-to-last dimensions, respectively.
        # Here, the feature dimension and the time step dimension are swapped.
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        # 将输入数据通过卷积层
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        # 应用 Sigmoid 激活函数
        x = self.act1(x)

        # se
        # 计算卷积层输出的平均值，用于后续的注意力机制
        # Calculate the mean of the output from the convolutional layer for the subsequent attention mechanism.
        avg = x.mean(dim=1)  # bs, window
        # 通过 se_fc 线性层计算注意力权重，并通过 softmax 函数进行归一化
        # Compute the attention weights through the se_fc linear layer
        # and normalize them using the softmax function.
        se_attn = self.se_fc(avg).softmax(dim=-1)  # bs, window
        # 应用注意力权重到卷积层的输出上
        # Apply the attention weights to the output of the convolutional layer.
        x = torch.einsum("bnd,bd->bnd", x, se_attn)

        # 应用最大池化层
        # Apply the max pooling layer.
        x = self.maxPool(x)  # bs, lstm_units, 1
        # 应用 Dropout 层
        x = self.drop(x)
        # 再次调整数据维度，以适应 LSTM 层的输入要求
        # Adjust the dimensions of the data again to meet the input requirements of the LSTM layer.
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        # 将数据通过 LSTM 层
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        # 应用 Tanh 激活函数
        x = self.act2(x)
        # 移除时间步维度，将数据从三维张量压缩为二维张量
        # Remove the time step dimension, compressing the data from a three-dimensional
        # tensor to a two-dimensional tensor.
        x = x.squeeze(dim=1)  # bs, 2*lstm_units
        # 通过线性层进行最终的预测
        # Pass the data through the linear layer for the final prediction.
        x = self.cls(x)
        # 应用最终的 Tanh 激活函数
        # Apply the final Tanh activation function.
        x = self.act4(x)
        return x


class CNNLSTMModel_CBAM(nn.Module):

    def __init__(self, window=15, dim=15, lstm_units=128, num_layers=1):
        super(CNNLSTMModel_CBAM, self).__init__()
        self.conv1d = nn.Conv1d(dim, lstm_units, 1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.02)
        self.lstm = nn.LSTM(lstm_units, lstm_units, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(lstm_units * 2, 1)
        self.act4 = nn.Tanh()

        self.se_fc = nn.Linear(window, window)
        self.hw_fc = nn.Linear(lstm_units, lstm_units)

    def forward(self, x):
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        x = self.act1(x)

        # chanal
        avg = x.mean(dim=1)  # bs, window
        se_attn = self.se_fc(avg).softmax(dim=-1)  # bs, window
        x = torch.einsum("bnd,bd->bnd", x, se_attn)

        # wh
        avg = x.mean(dim=2)  # bs, lstm_units
        hw_attn = self.hw_fc(avg).softmax(dim=-1)  # bs, lstm_units
        x = torch.einsum("bnd,bn->bnd", x, hw_attn)

        x = self.maxPool(x)  # bs, lstm_units, 1
        x = self.drop(x)
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        x = self.act2(x)
        x = x.squeeze(dim=1)  # bs, 2*lstm_units
        x = self.cls(x)
        x = self.act4(x)
        return x


class CNNLSTMModel_HW(nn.Module):

    def __init__(self, window=5, dim=6, lstm_units=64, num_layers=1):
        super(CNNLSTMModel_HW, self).__init__()
        self.conv1d = nn.Conv1d(dim, lstm_units, 1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.1)
        self.lstm = nn.LSTM(lstm_units, lstm_units, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(lstm_units * 2, 1)
        self.act4 = nn.Tanh()

        self.hw_fc = nn.Linear(lstm_units, lstm_units)

    def forward(self, x):
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        x = self.act1(x)

        # wh
        avg = x.mean(dim=2)  # bs, lstm_units
        hw_attn = self.hw_fc(avg).softmax(dim=-1)  # bs, lstm_units
        x = torch.einsum("bnd,bn->bnd", x, hw_attn)

        x = self.maxPool(x)  # bs, lstm_units, 1
        x = self.drop(x)
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        x = self.act2(x)
        x = x.squeeze(dim=1)  # bs, 2*lstm_units
        x = self.cls(x)
        x = self.act4(x)
        return x
