import math

import torch
from torch.nn import (
    Transformer,
    TransformerEncoder,
    Module,
    Sequential,
    Linear,
    GELU,
)
from torch import nn



########################################################################################################################
######################################### Modules & Helper Functions ###################################################
########################################################################################################################
def time_conv_Allx1(input_units):
    """Allx1 convolution over time (taking all input channels into account at each timestep"""
    return torch.nn.Conv1d(input_units, input_units, kernel_size=1, padding=0, groups=1)


def time_conv_1x3(input_units, depth="channelwise"):
    """1x3 convolution over time channelwise """
    if depth == "channelwise":
        groups = input_units
    elif depth == "full":
        groups = 1
    else:
        groups = depth

    return torch.nn.Conv1d(input_units, input_units, kernel_size=3, padding=1, groups=groups)


def time_conv_1x5(input_units, depth="channelwise"):
    """1x5 convolution over time channelwise """
    assert depth in ["channelwise", "full"], "depth can only be channelwise or full"
    if depth == "channelwise":
        groups = input_units
    elif depth == "full":
        groups = 1
    else:
        groups = depth
    return torch.nn.Conv1d(input_units, input_units, kernel_size=5, padding=2, groups=groups)

def add_vel_and_acc_info(x):
    """
     Approximate Velocity and Accelartion and add to feature dimension
    :param x: torch.Tensor
        3D-Tensor (batch, seq_length, channels)
    :return: torch.Tensor
        3D-Tensor (batch,seq_length, 3*channels)
    """
    zeros = torch.zeros(x.shape[0], 1, x.shape[2],device = x.device)#.to(DEVICE)
    velocity = x[:, 1:, :] - x[:, :-1, :]
    acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]
    velocity = torch.cat((velocity, zeros), axis=1)
    acceleration = torch.cat((zeros, acceleration, zeros), axis=1)
    x = torch.cat((x, velocity, acceleration), axis=2)
    return x

def double_sequence(x):
    """
    Interpolate between time steps
    :param x: torch.Tensor
        3D-Tensor (batch, seq_length, channels)
    :return: torch.Tensor
        3D-Tensor (batch, 2*seq_length, channels)
    """
    x1 = x
    x2 = (x[:, :-1, :] + x[:, 1:, :]) / 2.0
    x2 = torch.cat([x2, x1[:, -1, :].view(x1.shape[0], 1, x1.shape[2])], axis=1)

    #x = torch.zeros((x1.shape[0], x1.shape[1] + x2.shape[1], x1.shape[2]), dtype=torch.double, requires_grad=True, device=DEVICE)
    x = torch.zeros((x1.shape[0], x1.shape[1] + x2.shape[1], x1.shape[2]), dtype=torch.double,
                    requires_grad=False, device = x.device)
    x[:, ::2, :] = x1  # Index every second row, starting from 0
    x[:, 1::2, :] = x2

    return x

class TimeConvIncpetionBlock(torch.nn.Module):
    def __init__(self,input_units, pre_activation_function, add_resid=True):
        super(TimeConvIncpetionBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.band_conv1d_1 = time_conv_Allx1(input_units)
        self.band_conv1d_3 = time_conv_1x3(input_units)
        self.band_conv1d_5 = time_conv_1x5(input_units)
        self.band_conv1d_combine = torch.nn.Conv1d(3 * input_units, input_units, 1, padding=0, groups=input_units)

        self.activation = pre_activation_function
        self.add_resid = add_resid

    def forward(self, x):
        resid = x

        out = self.activation(x)
        out1 = self.band_conv1d_1(out)
        out3 = self.band_conv1d_3(out)
        out5 = self.band_conv1d_5(out)


        out = [torch.stack((out1[:, i, :], out3[:, i, :], out5[:, i, :]), axis=1) for i in range(out.shape[1])]
        out = torch.cat(out, axis=1)
        if self.add_resid:
            out += resid

        return out



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


class MelChannelConv1D(torch.nn.Module):
    def __init__(self, input_units, filter_size_channel):
        super(MelChannelConv1D, self).__init__()
        self.filter_size_channel = filter_size_channel
        assert input_units % filter_size_channel == 0, 'output_size has to devisible by %d' % filter_size_channel
        output_units = int(input_units // filter_size_channel)
        self.ConvLayers = torch.nn.ModuleList(
            [torch.nn.Conv1d(input_units, output_units, 5, padding=2, groups=output_units) for _ in
             range(filter_size_channel)])

    def forward(self, x):
        batch, mel, seq = x.shape
        xs = []
        for i in range(self.filter_size_channel - 2):
            x_i = torch.cat((torch.zeros(batch, i + 1, seq, device = x.device), x[:, :-(i + 1), :]), axis=1)
            xs.append(x_i)
        xs.append(x)
        xs.append(torch.cat((x[:, 1:, :], torch.zeros(batch, 1, seq, device = x.device)), axis=1))

        outputs = []
        for i, layer in enumerate(self.ConvLayers):
            output_i = layer(xs[i])
            outputs.append(output_i)

        output = [torch.stack([res[:, i, :] for res in outputs], axis=1) for i in range(output_i.shape[1])]
        output = torch.cat(output, axis=1)

        return output



########################################################################################################################
################################################# Inverse Models  ######################################################
########################################################################################################################

class InverseModelMelTimeSmoothResidual(torch.nn.Module):
    """
        InverseModel
        - Initial Conv1d Layers for Convolution over time and neighbouring Mel Channels with residual connections
        - stacked LSTM-Cells
        - Post Conv1d Layers for Convolution over time stacked with residual connections
        - Weighting of lstm output and smoothed output
    """

    def __init__(self, input_size=60,
                 output_size=30,
                 hidden_size=180,
                 num_lstm_layers=4,
                 mel_smooth_layers=3,
                 mel_smooth_filter_size=3,
                 mel_resid_activation = torch.nn.Identity(),
                 resid_blocks=5,
                 time_filter_size=5,
                 pre_resid_activation = torch.nn.Identity(),
                 post_resid_activation = torch.nn.Identity(),
                 output_activation = torch.nn.Identity(),
                 lstm_resid=True):
        super().__init__()

        self.lstm_resid = lstm_resid
        self.mel_resid_activation = mel_resid_activation
        self.pre_activation = pre_resid_activation
        self.post_activation = post_resid_activation
        self.output_activation = output_activation

        self.double_sequence = double_sequence
        self.add_vel_and_acc_info = add_vel_and_acc_info
        self.MelBlocks = torch.nn.ModuleList(
            [MelChannelConv1D(input_size, mel_smooth_filter_size) for _ in range(mel_smooth_layers)])
        self.lstm = torch.nn.LSTM(3 * (input_size), hidden_size, num_layers=num_lstm_layers, batch_first=True)
        self.post_linear = torch.nn.Linear(hidden_size, output_size)
        self.ResidualConvBlocks = torch.nn.ModuleList(
            [TimeConvResBlock(output_size, time_filter_size, self.pre_activation, self.post_activation) for _ in
             range(resid_blocks)])

        if self.lstm_resid and len(self.ResidualConvBlocks) > 0:
            self.resid_weighting = torch.nn.Conv1d(2 * output_size, output_size, time_filter_size, padding=2,
                                               groups=output_size)

    def forward(self, x, *args):
        if len(self.MelBlocks) > 0:
            x = x.permute(0, 2, 1)
            for layer in self.MelBlocks:
                shortcut = x
                x = layer(x)
                x += shortcut
                x = self.mel_resid_activation(x)
            x = x.permute(0, 2, 1)

        x = self.add_vel_and_acc_info(x)
        output, _ = self.lstm(x)
        output = self.post_linear(output)
        output = self.double_sequence(output)

        output = output.permute(0, 2, 1)
        lstm_output = output
        for layer in self.ResidualConvBlocks:
            output = layer(output)

        if len(self.ResidualConvBlocks) > 0 and self.lstm_resid:
            output = [torch.stack((output[:, i, :], lstm_output[:, i, :]), axis=1) for i in range(output.shape[1])]
            output = torch.cat(output, axis=1)
            output = self.resid_weighting(output)

        output = self.output_activation(output.permute(0, 2, 1))
        return output




########################################################################################################################
################################################# Forward Models  ######################################################
########################################################################################################################

class ForwardModelMelTimeSmoothResidual(torch.nn.Module):
    """
        ForwardModel
        - Initial Conv1d layers for Convolution over time stacked with residual connections
        - stacked LSTM-Cells
        - Post Conv1d Layers for Convolution over time and neighbouring Mel Channels with residual connections
        - Weighting of lstm output and smoothed output
    """

    def __init__(self, input_size=30,
                 output_size=60,
                 hidden_size=180,
                 num_lstm_layers=4,
                 mel_smooth_layers=3,
                 mel_smooth_filter_size=3,
                 mel_resid_activation = torch.nn.Identity(),
                 resid_blocks=5,
                 pre_resid_activation=torch.nn.Identity(),
                 post_resid_activation=torch.nn.Identity(),
                 time_filter_size=5,
                 lstm_resid = True,
                 output_activation = torch.nn.Identity()):
        super().__init__()

        self.lstm_resid = lstm_resid
        self.pre_activation = pre_resid_activation
        self.post_activation = post_resid_activation
        self.output_activation = output_activation
        self.mel_resid_activation = mel_resid_activation
        self.ResidualConvBlocks = torch.nn.ModuleList(
            [TimeConvResBlock(input_size, time_filter_size, self.pre_activation, self.post_activation) for _ in range(resid_blocks)])
        self.half_sequence = torch.nn.AvgPool1d(2, stride=2)
        self.add_vel_and_acc_info = add_vel_and_acc_info
        self.lstm = torch.nn.LSTM(3 * (input_size), hidden_size, num_layers=num_lstm_layers, batch_first = True)
        self.post_linear = torch.nn.Linear(hidden_size, output_size)
        self.MelBlocks = torch.nn.ModuleList(
            [MelChannelConv1D(output_size, mel_smooth_filter_size) for _ in range(mel_smooth_layers)])

        if self.lstm_resid and len(self.MelBlocks) >0:
            self.resid_weighting = torch.nn.Conv1d(2 * output_size, output_size, time_filter_size, padding=2,
                                                   groups=output_size)

    def forward(self, x,*args):
        if len(self.ResidualConvBlocks) >0:
            x = x.permute(0, 2, 1)
            for layer in self.ResidualConvBlocks:
                x = layer(x)
            x = x.permute(0, 2, 1)
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
            output = self.mel_resid_activation(output)

        if len(self.MelBlocks) >0 and self.lstm_resid:
            output = [torch.stack((lstm_output[:, i, :], output[:, i, :]), axis=1) for i in range(output.shape[1])]
            output = torch.cat(output, axis=1)
            output = self.resid_weighting(output)
        output = output.permute(0, 2, 1)
        output = self.output_activation(output)

        return output

class ForwardModel(torch.nn.Module):
    """
        ForwardModel
        - Initial Conv1d layers for Convolution over time stacked with residual connections
        - stacked LSTM-Cells
        - Post Conv1d Layers for Convolution over time and neighbouring Mel Channels with residual connections
        - Weighting of lstm output and smoothed output
    """

    def __init__(self, input_size=30,
                 output_size=60,
                 hidden_size=180,
                 num_lstm_layers=4,
                 apply_half_sequence=True):
        super().__init__()

        self.apply_half_sequence = apply_half_sequence
        if self.apply_half_sequence:
            self.half_sequence = torch.nn.AvgPool1d(2, stride=2)
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_lstm_layers, batch_first=True)
        self.post_linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, *args):
        output, _ = self.lstm(x)
        output = self.post_linear(output)
        if self.apply_half_sequence:
            output = output.permute(0, 2, 1)
            output = self.half_sequence(output)
            output = output.permute(0, 2, 1)

        return output

########################################################################################################################
############################################### Embbedder Models  ######################################################
########################################################################################################################

class MelEmbeddingModelMelSmoothResidualUpsampling(torch.nn.Module):
    """
        EmbedderModel
        - Initial Conv1d Layers for Convolution over time and neighbouring Mel Channels with residual connections
        - stacked LSTM-Cells
        - Post upsammpling layer
    """

    def __init__(self, input_size=60,
                 output_size=300,
                 hidden_size=180,
                 num_lstm_layers=4,
                 mel_smooth_layers=3,
                 mel_smooth_filter_size=3,
                 mel_resid_activation = torch.nn.Identity(),
                 post_activation = torch.nn.LeakyReLU(),
                 post_upsampling_size=8192):
        super().__init__()

        self.mel_resid_activation = mel_resid_activation
        # self.add_vel_and_acc_info = add_vel_and_acc_info
        self.MelBlocks = torch.nn.ModuleList(
            [MelChannelConv1D(input_size, mel_smooth_filter_size) for _ in range(mel_smooth_layers)])
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_lstm_layers, batch_first=True)
        self.post_linear = torch.nn.Linear(hidden_size, post_upsampling_size)
        self.upsampling = torch.nn.Linear(post_upsampling_size, output_size)
        self.post_activation = post_activation
        # self.resid_weighting = torch.nn.Conv1d(2 * output_size, output_size, time_filter_size, padding=2, groups=output_size)

    def forward(self, x, lens, *args):
        # IntermediateOutputs = []
        if len(self.MelBlocks) >0:
            x = x.permute(0, 2, 1)
            for layer in self.MelBlocks:
                shortcut = x
                x = layer(x)
                x += shortcut
                x = self.mel_resid_activation(x)

            x = x.permute(0, 2, 1)
        # x = self.add_vel_and_acc_info(x)
        output, (h_n, _) = self.lstm(x)
        output = torch.stack([output[i, (last - 1).long(), :] for i, last in enumerate(lens)])
        output = self.post_linear(output)
        output = self.post_activation(output)
        output = self.upsampling(output)

        return output



class EmbeddingModel(torch.nn.Module):
    """
        Embedder
        - Initial Conv1d Layers for Convolution over time and neighbouring Mel Channels with residual connections
        - stacked LSTM-Cells
        - Post upsammpling layer
    """

    def __init__(self, input_size=60,
                 output_size=300,
                 hidden_size=720,
                 num_lstm_layers=1,
                 post_activation = torch.nn.LeakyReLU(),
                 post_upsampling_size=0,
                 dropout=0):
        super().__init__()
        
        self.post_upsampling_size = post_upsampling_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_lstm_layers, batch_first=True, dropout = dropout)
        if post_upsampling_size >0:
            self.post_linear = torch.nn.Linear(hidden_size, post_upsampling_size)
            self.linear_mapping = torch.nn.Linear(post_upsampling_size, output_size)
            self.post_activation = post_activation
        else:
            self.linear_mapping = torch.nn.Linear(hidden_size, output_size)
    

    def forward(self, x, lens, *args):
        output, (h_n, _) = self.lstm(x)
        output = torch.stack([output[i, (last - 1).long(), :] for i, last in enumerate(lens)])
        if self.post_upsampling_size>0:
            output = self.post_linear(output)
            output = self.post_activation(output)
        output = self.linear_mapping(output)

        return output


########################################################################################################################
################################################# Baseline Models  #####################################################
########################################################################################################################

class LinearModel(torch.nn.Module):
    def __init__(self,
                 input_channel = 30,
                 output_channel = 60,
                 mode = "inv",
                 on_full_sequence = False,
                 add_vel_and_acc=True):
        super().__init__()
        self.on_full_sequence = on_full_sequence
        self.add_vel_and_acc = add_vel_and_acc
        assert mode in ["pred", "inv", "embed"], "if you want to train a predictive model please set mode to 'pred', for a inverse model set mode to 'inv'!"
        self.mode = mode
        if self.on_full_sequence:
            if self.add_vel_and_acc:
                self.input_channel = 3 * input_channel
                self.add_vel_and_acc_info = add_vel_and_acc_info
            else:
                self.input_channel = input_channel
            if self.mode == "pred":
                self.half_sequence = torch.nn.AvgPool1d(2, stride=2)
            elif self.mode == "inv":
                self.double_sequence = double_sequence

        else:
            self.input_channel = 2*input_channel
        self.output_channel = output_channel
        self.linear = torch.nn.Linear(self.input_channel, self.output_channel)


    def forward(self, x, *args):
        if self.on_full_sequence:
            if self.add_vel_and_acc:
                x = self.add_vel_and_acc_info(x)
        else:
            x = x.reshape((x.shape[0],1,-1))

        output = self.linear(x)
        if self.on_full_sequence:
            if self.mode == "pred":
                output = output.permute(0, 2, 1)
                output = self.half_sequence(output)
                output = output.permute(0, 2, 1)
            elif self.mode == "inv":
                output = self.double_sequence(output)
        return output


class NonLinearModel(torch.nn.Module):
    def __init__(self,
                 input_channel=30,
                 output_channel=60,
                 hidden_units=8192,
                 activation_function=torch.nn.LeakyReLU(),
                 mode = "pred",
                 on_full_sequence=False,
                 add_vel_and_acc=True):
        super().__init__()
        self.on_full_sequence = on_full_sequence
        self.add_vel_and_acc = add_vel_and_acc
        assert mode in ["pred", "inv", "embed"], "if you want to train a predictive model please set mode to 'pred', for a inverse model set mode to 'inv'!"
        self.mode = mode
        if self.on_full_sequence:
            if self.add_vel_and_acc:
                self.input_channel = input_channel * 3
                self.add_vel_and_acc_info = add_vel_and_acc_info
            else:
                self.input_channel = input_channel
            if self.mode == "pred":
                self.half_sequence = torch.nn.AvgPool1d(2, stride=2)
            elif self.mode == "inv":
                self.double_sequence = double_sequence
        else:
            self.input_channel = input_channel * 2

        self.output_channel = output_channel
        self.hidden_units = hidden_units
        self.activation_function = activation_function
        self.non_linear = torch.nn.Linear(self.input_channel, self.hidden_units)
        self.linear = torch.nn.Linear(self.hidden_units, self.output_channel)

    def forward(self, x, *args):
        if self.on_full_sequence:
            if self.add_vel_and_acc:
                x = self.add_vel_and_acc_info(x)
            if self.mode == "embed":
                x = torch.sum(x, axis=1)
        else:
            x = x.reshape((x.shape[0],1, -1))
        output = self.non_linear(x)
        output = self.activation_function(output)
        output = self.linear(output)
        if self.on_full_sequence:
            if self.mode == "pred":
                output = output.permute(0, 2, 1)
                output = self.half_sequence(output)
                output = output.permute(0, 2, 1)
            elif self.mode == "inv":
                output = self.double_sequence(output)
        return output

########################################################################################################################
############################################### Generative Models  #####################################################
########################################################################################################################

class Critic(torch.nn.Module):
    def __init__(self, input_size=30,
                 embed_size=300,
                 hidden_size=180,
                 num_res_blocks=5):
        super().__init__()

        self.inital_linear = torch.nn.Linear(input_size + embed_size, hidden_size)
        self.res_blocks = torch.nn.ModuleList(
            [self._block(hidden_size, hidden_size, 5, 1, 2) for _ in range(num_res_blocks)])

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels, out_channels, kernel_size, stride, padding,
            ),
            torch.nn.InstanceNorm1d(out_channels, affine=True),
            torch.nn.LeakyReLU(0.2),
        )

    def forward(self, x, length, vector):
        x = torch.cat([x, vector.unsqueeze(1).repeat(1, x.shape[1], 1)], dim=2)
        output = self.inital_linear(x)
        output = output.permute(0, 2, 1)

        for i, block in enumerate(self.res_blocks):
            resid = output
            output = block(output)
            output += resid

            # average pooling
        output = output.mean([1, 2])
        return output


class Generator(torch.nn.Module):
    def __init__(self, channel_noise=100,
                 embed_size=300,
                 fc_size=1024,
                 inital_seq_length=4,
                 hidden_size=256,
                 num_res_blocks=5,
                 output_size=30):
        super().__init__()

        self.fc_size = fc_size
        self.hidden_size = hidden_size
        self.fc_reshaped_size = int(fc_size / inital_seq_length)
        self.fully_connected = torch.nn.Linear(channel_noise + embed_size, fc_size)

        self.res_blocks = torch.nn.ModuleList([self._block(self.fc_reshaped_size, hidden_size, 5, 1, 2)])
        self.res_blocks = self.res_blocks.extend(
            torch.nn.ModuleList([self._block(hidden_size, hidden_size, 5, 1, 2) for _ in range(num_res_blocks - 1)]))

        self.post_linear = torch.nn.Linear(hidden_size, output_size)
        self.final_smoothing = torch.nn.Conv1d(output_size, output_size, kernel_size=5, padding=2, groups=output_size)
        self.output_activation = torch.nn.Tanh()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            ),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.LeakyReLU(0.2),
        )

    def forward(self, x, length, vector):
        x = torch.cat([x, vector.unsqueeze(1)], dim=2)
        output = self.fully_connected(x)
        output = output.view((len(x), self.fc_reshaped_size, int(output.shape[-1] / self.fc_reshaped_size)))

        for i, block in enumerate(self.res_blocks):
            size_ = int(length / (len(self.res_blocks) - i))
            resizing = torch.nn.Upsample(size=(size_), mode='linear', align_corners=False)
            output = resizing(output)
            resid = output
            output = block(output)
            if i == 0:
                if self.fc_reshaped_size == self.hidden_size:
                    output += resid
            else:
                output += resid

        output = output.permute(0, 2, 1)
        output = self.post_linear(output)
        output = output.permute(0, 2, 1)
        resid = output
        output = self.final_smoothing(output)
        output += resid
        output = output.permute(0, 2, 1)
        output = self.output_activation(output)

        return output


class SemVecToCpModel(torch.nn.Module):
    def __init__(self,
                 input_size=300, #semantic vector dim
                 output_size=30,
                 hidden_size=180,
                 num_lstm_layers=4,
                 resid_blocks=5,
                 time_filter_size=5,
                 pre_resid_activation = torch.nn.Identity(),
                 post_resid_activation = torch.nn.Identity(),
                 output_activation = torch.nn.Identity(),
                 lstm_resid=True):
        super().__init__()

        self.lstm_resid = lstm_resid
        self.pre_activation = pre_resid_activation
        self.post_activation = post_resid_activation
        self.output_activation = output_activation

        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_lstm_layers, batch_first=True)
        self.post_linear = torch.nn.Linear(hidden_size, output_size)
        self.ResidualConvBlocks = torch.nn.ModuleList(
            [TimeConvResBlock(output_size, time_filter_size, self.pre_activation, self.post_activation) for _ in
             range(resid_blocks)])
        if self.lstm_resid and len(self.ResidualConvBlocks) > 0:
            self.resid_weighting = torch.nn.Conv1d(2 * output_size, output_size, time_filter_size, padding=2,
                                               groups=output_size)

    def forward(self,x, *args):
        output, _ = self.lstm(x)
        output = self.post_linear(output)

        output = output.permute(0, 2, 1)
        lstm_output = output
        for layer in self.ResidualConvBlocks:
            output = layer(output)

        if len(self.ResidualConvBlocks) > 0 and self.lstm_resid:
            output = [torch.stack((output[:, i, :], lstm_output[:, i, :]), axis=1) for i in range(output.shape[1])]
            output = torch.cat(output, axis=1)
            output = self.resid_weighting(output)

        output = self.output_activation(output.permute(0, 2, 1))
        return output



class SemVecToMelModel(torch.nn.Module):
    def __init__(self,
                 input_size=300,  # semantic vector dim
                 output_size=60,
                 hidden_size=180,
                 num_lstm_layers=4,
                 mel_smooth_layers=3,
                 mel_smooth_filter_size=3,
                 mel_resid_activation=torch.nn.Identity(),
                 time_filter_size=5,
                 output_activation=torch.nn.Identity(),
                 lstm_resid=True):
        super().__init__()

        self.lstm_resid = lstm_resid
        self.output_activation = output_activation
        self.mel_resid_activation = mel_resid_activation

        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_lstm_layers, batch_first=True)
        self.post_linear = torch.nn.Linear(hidden_size, output_size)
        self.MelBlocks = torch.nn.ModuleList(
            [MelChannelConv1D(output_size, mel_smooth_filter_size) for _ in range(mel_smooth_layers)])
        if self.lstm_resid and len(self.MelBlocks) > 0:
            self.resid_weighting = torch.nn.Conv1d(2 * output_size, output_size, time_filter_size, padding=2,
                                                   groups=output_size)

    def forward(self, x, *args):
        output, _ = self.lstm(x)
        output = self.post_linear(output)

        output = output.permute(0, 2, 1)
        lstm_output = output

        for layer in self.MelBlocks:
            shortcut = output
            output = layer(output)
            output += shortcut
            output = self.mel_resid_activation(output)

        if len(self.MelBlocks) > 0 and self.lstm_resid:
            output = [torch.stack((lstm_output[:, i, :], output[:, i, :]), axis=1) for i in range(output.shape[1])]
            output = torch.cat(output, axis=1)
            output = self.resid_weighting(output)

        output = self.output_activation(output.permute(0, 2, 1))
        return output


class LSTMCritic(torch.nn.Module):
    def __init__(self, input_size=30,
                 embed_size = 300, 
                 output_size=1,
                 hidden_size=200,
                 num_lstm_layers=2,
                 dropout=0.5):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size + embed_size, hidden_size, num_layers=num_lstm_layers, batch_first = True, dropout=dropout)
        self.fully_connected = torch.nn.Linear(hidden_size,output_size)

    def forward(self, x, lens, vector, *args):
        
        x = torch.cat([x, vector.unsqueeze(1).repeat(1, x.shape[1], 1)],dim=2)
        output, (h_n, _) = self.lstm(x)
        output = torch.stack([output[i, (last - 1).long(), :] for i, last in enumerate(lens)])
        output = self.fully_connected(output)
        # average pooling
        #output = output.mean([1])
        
        return output


class LSTMGenerator(torch.nn.Module):
    def __init__(self,channel_noise = 60,
                 embed_size = 300,
                 output_size=30,
                 hidden_size=200,
                 num_lstm_layers=2,
                 dropout=0.5,
                 activation = torch.nn.LeakyReLU(0.2)):
        super().__init__()
        
        self.output_activation = torch.nn.Tanh()
        self.activation = activation
        self.fully_connected = torch.nn.Linear(channel_noise + embed_size, hidden_size)
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size, num_layers=num_lstm_layers, batch_first = True, dropout=dropout)
        self.post_linear = torch.nn.Linear(hidden_size, output_size)


    def forward(self, x, lens, vector, *args):
        x = torch.cat([x,vector.unsqueeze(1).repeat(1,x.shape[1],1)], dim = 2)
        output = self.fully_connected(x)
        output = self.activation(output)
        output, _ = self.lstm(output)
        #output = torch.stack([output[i, (last - 1).long(), :] for i, last in enumerate(lens)])
        
        output = self.post_linear(output)
        output = self.output_activation(output)

        return output



class PositionalEncoding(Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        position = torch.arange(0, max_len)
        position = position.unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, input_):
        output = input_ + self.pe[:, : input_.size(1), :]
        return output


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=GELU()
    ):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None):
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class SpeechNonSpeechTransformer(Transformer):
    def __init__(self, input_dim, num_layers, nhead, output_dim):
        super().__init__()

        self.pos_encoder = PositionalEncoding(input_dim, dropout=0.1)
        self.encoder_layer = CustomTransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=1024, activation=GELU()
        )
        self.transformer_encoder = TransformerEncoder(
            self.encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.linear = Sequential(Linear(input_dim, 20), GELU(), Linear(20, 1))


    def forward(self, input_, *, src_lens=None):
        mask = torch.zeros((input_.size(0), input_.size(1)), device=input_.device, dtype=input_.dtype)
        if src_lens is not None:
            for ii, len_ in enumerate(src_lens):
                mask[ii, len_:] = float("-inf")

        output = self.pos_encoder(input_)
        output = self.transformer_encoder(output, src_key_padding_mask=mask)
        output = output.mean(dim=1)
        output = self.linear(output)

        output = torch.squeeze(output, 1)

        return output


class LinearClassifier(Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.linear = Linear(input_dim, output_dim)


    def forward(self, input_, *, src_lens=None):

        output = self.linear(input_)
        output = torch.squeeze(output, 2)

        # set all padded values to zero
        if src_lens is not None:
            for ii, len_ in enumerate(src_lens):
                output[ii, len_:] = 0.0

        # sum all values and devide by seq length
        if src_lens is not None:
            output = output.sum(dim=1) / torch.tensor(src_lens, device=output.device)
        else:
            output = output.mean(dim=1)

        return output

