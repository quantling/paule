import torch


def add_vel_and_acc_info(x):
    """
     Approximate Velocity and Accelartion and add to feature dimension
    :param x: torch.Tensor
        3D-Tensor (batch, seq_length, channels)
    :return: torch.Tensor
        3D-Tensor (batch,seq_length, 3*channels)
    """
    zeros = torch.zeros(x.shape[0], 1, x.shape[2]).to(DEVICE)
    velocity = x[:, 1:, :] - x[:, :-1, :]
    acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]
    velocity = torch.cat((velocity, zeros), axis=1)
    acceleration = torch.cat((zeros, acceleration, zeros), axis=1)
    x = torch.cat((x, velocity, acceleration), axis=2)
    return x


def time_conv_Allx1(input_units):
    """Allx1 convolution over time (taking all input channels into account at each timestep"""
    return torch.nn.Conv1d(input_units, input_units,kernel_size = 1, padding=0, groups=1)

def time_conv_1x3(input_units, depth = "channelwise"):
    """1x3 convolution over time channelwise """
    if depth == "channelwise":
        groups = input_units
    elif depth == "full":
        groups = 1
    else:
        groups = depth

    return torch.nn.Conv1d(input_units, input_units,kernel_size = 3, padding=1, groups=groups)

def time_conv_1x5(input_units, depth = "channelwise"):
    """1x5 convolution over time channelwise """
    assert depth in ["channelwise", "full"], "depth can only be channelwise or full"
    if depth == "channelwise":
        groups = input_units
    elif depth == "full":
        groups = 1
    else:
        groups = depth
    return torch.nn.Conv1d(input_units, input_units,kernel_size = 5, padding=2, groups=groups)

class MelChannelConv1D(torch.nn.Module):
    def __init__(self,input_units,filter_size_channel):
        super(MelChannelConv1D, self).__init__()
        self.filter_size_channel = filter_size_channel
        assert input_units % filter_size_channel == 0, 'output_size has to devisible by %d' % filter_size_channel
        output_units = int(input_units // filter_size_channel)
        self.ConvLayers = torch.nn.ModuleList([torch.nn.Conv1d(input_units, output_units, 5, padding=2, groups=output_units) for _ in range(filter_size_channel)])

    def forward(self,x):
        batch, mel, seq = x.shape
        xs = []
        for i in range(self.filter_size_channel -2):
            x_i = torch.cat((torch.zeros(batch, i+1, seq).to(DEVICE), x[:, :-(i+1),:]), axis=1)

            xs.append(x_i)
        xs.append(x)
        xs.append(torch.cat((x[:,1:,:], torch.zeros(batch, 1, seq).to(DEVICE)), axis=1))

        outputs = []
        for i,layer in enumerate(self.ConvLayers):
            output_i = layer(xs[i])
            outputs.append(output_i)

        output = [torch.stack([res[:, i, :] for res in outputs], axis=1) for i in range(output_i.shape[1])]
        output = torch.cat(output, axis = 1)

        return output

class TimeConvResBlock(torch.nn.Module):
    def __init__(self,input_units,filter_size,pre_activation_function = torch.nn.Identity(),
                 post_activation_function = torch.nn.Identity(),
                 add_resid=True, depth="channelwise"):
        super(TimeConvResBlock, self).__init__()
        assert filter_size in [5, 3], "should be a valid filter size (5 or 3)"
        if filter_size == 3:
            self.band_conv1d_1 = time_conv_1x3(input_units,depth)
            self.band_conv1d_2 = time_conv_1x3(input_units,depth)
        else:
            self.band_conv1d_1 = time_conv_1x5(input_units,depth)
            self.band_conv1d_2 = time_conv_1x5(input_units,depth)

        self.pre_activation = pre_activation_function
        self.post_activation = post_activation_function
        self.add_resid = add_resid

    def forward(self,x):
        resid = x

        out = self.band_conv1d_1(self.pre_activation(x))
        out = self.band_conv1d_2(out)
        out = self.post_activation(out)
        if self.add_resid:
            out += resid
        return out


class ForwardModel_ResidualSmoothMel(torch.nn.Module):
    """
        ForwardModel with time smoothing and convolution over channels"
    """

    def __init__(self, input_size=30,
                 output_size=60,
                 hidden_size=180,
                 num_lstm_layers=4,
                 n_blocks = 3,
                 mel_smooth_layers = 3,
                 mel_smooth_filter_size = 3,
                 resiudal_blocks=5, time_filter_size = 5):
        super().__init__()

        self.activation = torch.nn.LeakyReLU()
        self.ResidualConvBlocks = torch.nn.ModuleList([TimeConvResBlock(input_size,time_filter_size) for _ in range(resiudal_blocks)])
        self.half_sequence = torch.nn.AvgPool1d(2, stride=2)
        self.add_vel_and_acc_info = add_vel_and_acc_info
        self.lstm = torch.nn.LSTM(3 * (input_size), hidden_size, num_layers=num_lstm_layers, batch_first = True)
        self.post_linear = torch.nn.Linear(hidden_size, output_size)
        self.MelBlocks = torch.nn.ModuleList([MelChannelConv1D(output_size,mel_smooth_filter_size) for _ in range(mel_smooth_layers)])
        self.block_activation = torch.nn.LeakyReLU()
        self.resid_weighting = torch.nn.Conv1d(2 * output_size, output_size, time_filter_size, padding=2, groups=output_size)

    def forward(self,x):
        x = x.permute(0,2,1)
        for layer in self.ResidualConvBlocks:
            x = layer(x)
        x = x.permute(0,2,1)
        x = self.add_vel_and_acc_info(x)
        output, _ = self.lstm(x)
        output = self.post_linear(output)
        output = output.permute(0, 2, 1)
        output = self.half_sequence(output)
        lstm_output = output

        for layer in self.MelBlocks:
            shortcut = output
            output = layer(output)
            output += shortcut
            output = self.block_activation(output)

        output = [torch.stack((lstm_output[:, i, :], output[:, i, :]), axis=1) for i in range(output.shape[1])]
        output = torch.cat(output, axis=1)
        output = self.resid_weighting(output)
        output = output.permute(0, 2, 1)
        output = self.activation(output)

        return output


def double_sequence(output):
    # by Tino
    # double seq
    output1 = output
    output2 = (output[:, :-1, :] + output[:, 1:, :]) / 2.0
    output2 = torch.cat([output2, output1[:, -1, :].view(output1.shape[0], 1, output1.shape[2])], axis=1)
    output = torch.zeros((output1.shape[0],output1.shape[1] + output2.shape[1], output1.shape[2]), dtype=torch.double, requires_grad = True)  #.to(device)
    output[:, ::2, :] = output1   # Index every second row, starting from 0
    output[:, 1::2, :] = output2
    return output



#def double_sequence(x):
#   """
#   Interpolate between time steps
#   :param x: torch.Tensor
#       3D-Tensor (batch, seq_length, channels)
#   :return: torch.Tensor
#       3D-Tensor (batch, 2*seq_length, channels)
#   """
#   x1 = x
#   x2 = (x[:, :-1, :] + x[:, 1:, :]) / 2.0
#   x2 = torch.cat([x2, x1[:, -1, :].view(x1.shape[0], 1, x1.shape[2])], axis=1)
#
#   x = torch.zeros((x1.shape[0], x1.shape[1] + x2.shape[1], x1.shape[2]), dtype=torch.double, requires_grad=True).cuda()
#   x[:, ::2, :] = x1  # Index every second row, starting from 0
#   x[:, 1::2, :] = x2
#
#   return x


class InverseModel_MelSmoothResidual(torch.nn.Module):
    """
        InverseModel with Inital Conv1d Layers for Convolution over time and Mel Channels + Post Resnet like modules  for smoothing over time"
    """

    def __init__(self, input_size=60,
                 output_size=30,
                 hidden_size=180,
                 num_lstm_layers=4,
                 mel_smooth_layers = 3,
                 mel_smooth_filter_size = 3,
                 resiudal_blocks=5, time_filter_size = 5):
        super().__init__()

        self.pre_activation = torch.nn.Identity()
        self.post_activation = torch.nn.Identity()
        self.block_activation = torch.nn.LeakyReLU()
        self.double_sequence = double_sequence
        self.add_vel_and_acc_info = add_vel_and_acc_info
        self.MelBlocks = torch.nn.ModuleList([MelChannelConv1D(input_size,mel_smooth_filter_size) for _ in range(mel_smooth_layers)])
        self.lstm = torch.nn.LSTM(3 * (input_size), hidden_size, num_layers=num_lstm_layers, batch_first = True)
        self.post_linear = torch.nn.Linear(hidden_size, output_size)
        self.ResidualConvBlocks = torch.nn.ModuleList([TimeConvResBlock(output_size,time_filter_size, self.pre_activation,self.post_activation) for _ in range(resiudal_blocks)])
        self.resid_weighting = torch.nn.Conv1d(2 * output_size, output_size, time_filter_size, padding=2, groups=output_size)


    def forward(self, x):
        #IntermediateOutputs = []
        x = x.permute(0,2,1)
        for layer in self.MelBlocks:
            shortcut = x
            x = layer(x)
            x += shortcut
            x = self.block_activation(x)

        x = x.permute(0,2,1)
        x = self.add_vel_and_acc_info(x)
        output, _ = self.lstm(x)
        output = self.post_linear(output)
        output = self.double_sequence(output)

        output = output.permute(0, 2, 1)
        #IntermediateOutputs.append(output)
        lstm_output = output
        for layer in self.ResidualConvBlocks:
            output = layer(output)
            #IntermediateOutputs.append(output)


        #output = [torch.stack([res[:, i, :] for res in self.IntermediateOutputs], axis = 1) for i in range(output.shape[1])]
        output = [torch.stack((output[:, i, :],lstm_output[:, i, :]), axis = 1) for i in range(output.shape[1])]
        output = torch.cat(output, axis=1)
        output = self.resid_weighting(output)

        return output.permute(0, 2, 1)


class EmbeddingClassifierModel(torch.nn.Module):
    """
        Wide Embedding Model based on LSTM's with a post upsampling layer
    """
    def __init__(self, input_size=60, hidden_size1=800, num_lstm_layers1=2, post_size=8192, output_size=300):
        super().__init__()

        self.lstm1 = torch.nn.LSTM(input_size, hidden_size1, num_layers=num_lstm_layers1, batch_first=True)
        self.post_linear1 = torch.nn.Linear(hidden_size1, post_size)
        self.post_linear2 = torch.nn.Linear(post_size, output_size)
        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, input_, lens):
        output, (h_n, _) = self.lstm1(input_)
        output = torch.stack([output[i, (last - 1).long(), :] for i, last in enumerate(lens)])
        output = self.post_linear1(output)
        output = self.leaky_relu(output)
        output = self.post_linear2(output)

        return output



