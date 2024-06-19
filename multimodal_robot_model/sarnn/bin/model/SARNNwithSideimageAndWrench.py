#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import torch.nn as nn
from eipl.layer import SpatialSoftmax, InverseSpatialSoftmax


class SARNNwithSideimageAndWrench(nn.Module):
    #:: SARNNwithSideimageAndWrench
    """SARNN: Spatial Attention with Recurrent Neural Network.
    This model "explicitly" extracts positions from the image that are important to the task, such as the work object or arm position,
    and learns the time-series relationship between these positions and the robot's joint angles and wrench angles.
    The robot is able to generate robust motions in response to changes in object position and lighting.

    Arguments:
        rec_dim (int): The dimension of the recurrent state in the LSTM cell.
        k_dim (int, optional): The dimension of the attention points.
        joint_dim (int, optional): The dimension of the joint angles.
        wrench_dim (int, optional): The dimension of the wrench angles.
        temperature (float, optional): The temperature parameter for the softmax function.
        heatmap_size (float, optional): The size of the heatmap in the InverseSpatialSoftmax layer.
        kernel_size (int, optional): The size of the convolutional kernel.
        activation (str, optional): The name of activation function.
        im_size (list, optional): The size of the input image [height, width].
    """

    def __init__(
        self,
        rec_dim,
        k_dim=5,
        joint_dim=14,
        wrench_dim=14,
        temperature=1e-4,
        heatmap_size=0.1,
        kernel_size=3,
        im_size=[64, 64],
    ):
        super(SARNNwithSideimageAndWrench, self).__init__()

        self.k_dim = k_dim
        activation = nn.LeakyReLU(negative_slope=0.3)

        sub_im_size = [
            im_size[0] - 3 * (kernel_size - 1),
            im_size[1] - 3 * (kernel_size - 1),
        ]
        self.temperature = temperature
        self.heatmap_size = heatmap_size

        # Front Positional Encoder
        self.front_pos_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 0),  # Convolutional layer 1
            activation,
            nn.Conv2d(16, 32, 3, 1, 0),  # Convolutional layer 2
            activation,
            nn.Conv2d(32, self.k_dim, 3, 1, 0),  # Convolutional layer 3
            activation,
            SpatialSoftmax(
                width=sub_im_size[0],
                height=sub_im_size[1],
                temperature=self.temperature,
                normalized=True,
            ),  # Spatial Softmax layer
        )

        # Side Positional Encoder
        self.side_pos_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 0),  # Convolutional layer 1
            activation,
            nn.Conv2d(16, 32, 3, 1, 0),  # Convolutional layer 2
            activation,
            nn.Conv2d(32, self.k_dim, 3, 1, 0),  # Convolutional layer 3
            activation,
            SpatialSoftmax(
                width=sub_im_size[0],
                height=sub_im_size[1],
                temperature=self.temperature,
                normalized=True,
            ),  # Spatial Softmax layer
        )

        # Front Image Encoder
        self.front_im_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 0),  # Convolutional layer 1
            activation,
            nn.Conv2d(16, 32, 3, 1, 0),  # Convolutional layer 2
            activation,
            nn.Conv2d(32, self.k_dim, 3, 1, 0),  # Convolutional layer 3
            activation,
        )

        # Side Image Encoder
        self.side_im_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 0),  # Convolutional layer 1
            activation,
            nn.Conv2d(16, 32, 3, 1, 0),  # Convolutional layer 2
            activation,
            nn.Conv2d(32, self.k_dim, 3, 1, 0),  # Convolutional layer 3
            activation,
        )

        rec_in = joint_dim + wrench_dim + self.k_dim * 4
        self.rec = nn.LSTMCell(rec_in, rec_dim)  # LSTM cell

        # Joint Decoder
        self.decoder_joint = nn.Sequential(
            nn.Linear(rec_dim, joint_dim), activation
        )  # Linear layer and activation

        # Wrench Decoder
        self.decoder_wrench = nn.Sequential(
            nn.Linear(rec_dim, wrench_dim), activation
        )  # Linear layer and activation

        # Front Point Decoder
        self.decoder_front_point = nn.Sequential(
            nn.Linear(rec_dim, self.k_dim * 2), activation
        )  # Linear layer and activation

        # Side Point Decoder
        self.decoder_side_point = nn.Sequential(
            nn.Linear(rec_dim, self.k_dim * 2), activation
        )  # Linear layer and activation

        # Front Inverse Spatial Softmax
        self.front_issm = InverseSpatialSoftmax(
            width=sub_im_size[0],
            height=sub_im_size[1],
            heatmap_size=self.heatmap_size,
            normalized=True,
        )

        # Side Inverse Spatial Softmax
        self.side_issm = InverseSpatialSoftmax(
            width=sub_im_size[0],
            height=sub_im_size[1],
            heatmap_size=self.heatmap_size,
            normalized=True,
        )

        # Front Image Decoder
        self.decoder_front_image = nn.Sequential(
            nn.ConvTranspose2d(
                self.k_dim, 32, 3, 1, 0
            ),  # Transposed Convolutional layer 1
            activation,
            nn.ConvTranspose2d(32, 16, 3, 1, 0),  # Transposed Convolutional layer 2
            activation,
            nn.ConvTranspose2d(16, 3, 3, 1, 0),  # Transposed Convolutional layer 3
            activation,
        )

        # Side Image Decoder
        self.decoder_side_image = nn.Sequential(
            nn.ConvTranspose2d(
                self.k_dim, 32, 3, 1, 0
            ),  # Transposed Convolutional layer 1
            activation,
            nn.ConvTranspose2d(32, 16, 3, 1, 0),  # Transposed Convolutional layer 2
            activation,
            nn.ConvTranspose2d(16, 3, 3, 1, 0),  # Transposed Convolutional layer 3
            activation,
        )

        self.apply(self._weights_init)

    def _weights_init(self, m):
        """
        Tensorflow/Keras-like initialization
        """
        if isinstance(m, nn.LSTMCell):
            nn.init.xavier_uniform_(m.weight_ih)
            nn.init.orthogonal_(m.weight_hh)
            nn.init.zeros_(m.bias_ih)
            nn.init.zeros_(m.bias_hh)

        if (
            isinstance(m, nn.Conv2d)
            or isinstance(m, nn.ConvTranspose2d)
            or isinstance(m, nn.Linear)
        ):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, xif, xis, xvj, xvw, state=None):
        """
        Forward pass of the SARNN module.
        Predicts the front image, side image, joint angle, wrench angle, and attention at the next time based on the front image, image, joint angle, and wrench angle at time t.
        Predict the front image, side image, joint angles, wrench angles, and attention points for the next state (t+1) based on
        the front image, side image, joint angles and wrench angles of the current state (t).
        By inputting the predicted joint angles and wrench angles as control commands for the robot,
        it is possible to generate sequential motion based on sensor information.

        Arguments:
            xif (torch.Tensor): Input front image tensor of shape (batch_size, channels, height, width).
            xis (torch.Tensor): Input side image tensor of shape (batch_size, channels, height, width).
            xvj (torch.Tensor): Input joint vector tensor of shape (batch_size, input_dim).
            xvw (torch.Tensor): Input wrench vector tensor of shape (batch_size, input_dim).
            state (tuple, optional): Initial hidden state and cell state of the LSTM cell.

        Returns:
            y_front_image (torch.Tensor): Decoded front image tensor of shape (batch_size, channels, height, width).
            y_side_image (torch.Tensor): Decoded side image tensor of shape (batch_size, channels, height, width).
            y_joint (torch.Tensor): Decoded joint prediction tensor of shape (batch_size, joint_dim).
            y_wrench (torch.Tensor): Decoded wrench prediction tensor of shape (batch_size, wrench_dim).
            enc_front_pts (torch.Tensor): Encoded front points tensor of shape (batch_size, k_dim * 2).
            enc_side_pts (torch.Tensor): Encoded side points tensor of shape (batch_size, k_dim * 2).
            dec_front_pts (torch.Tensor): Decoded front points tensor of shape (batch_size, k_dim * 2).
            dec_side_pts (torch.Tensor): Decoded side points tensor of shape (batch_size, k_dim * 2).
            rnn_hid (tuple): Tuple containing the hidden state and cell state of the LSTM cell.
        """

        # Encode input front image
        front_im_hid = self.front_im_encoder(xif)
        enc_front_pts, _ = self.front_pos_encoder(xif)

        # Encode input side image
        side_im_hid = self.side_im_encoder(xis)
        enc_side_pts, _ = self.side_pos_encoder(xis)

        # Reshape encoded points and concatenate with input vector
        enc_front_pts = enc_front_pts.reshape(-1, self.k_dim * 2)
        enc_side_pts = enc_side_pts.reshape(-1, self.k_dim * 2)
        hid = torch.cat([enc_front_pts, enc_side_pts, xvj, xvw], -1)

        rnn_hid = self.rec(hid, state)  # LSTM forward pass
        y_joint = self.decoder_joint(rnn_hid[0])  # Decode joint prediction
        y_wrench = self.decoder_wrench(rnn_hid[0])  # Decode wrench prediction
        dec_front_pts = self.decoder_front_point(rnn_hid[0])  # Decode front points
        dec_side_pts = self.decoder_side_point(rnn_hid[0])  # Decode side points

        # Reshape decoded front points
        dec_front_pts_in = dec_front_pts.reshape(-1, self.k_dim, 2)
        front_heatmap = self.front_issm(dec_front_pts_in)  # Inverse Spatial Softmax
        front_hid = torch.mul(front_heatmap, front_im_hid)  # Multiply heatmap with front image feature `front_im_hid`

        # Reshape decoded side points
        dec_side_pts_in = dec_side_pts.reshape(-1, self.k_dim, 2)
        side_heatmap = self.side_issm(dec_side_pts_in)  # Inverse Spatial Softmax
        side_hid = torch.mul(side_heatmap, side_im_hid)  # Multiply heatmap with side image feature `side_im_hid`

        y_front_image = self.decoder_front_image(front_hid)  # Decode front image
        y_side_image = self.decoder_side_image(side_hid)  # Decode side image
        return y_front_image, y_side_image, y_joint, y_wrench, enc_front_pts, enc_side_pts, dec_front_pts, dec_side_pts, rnn_hid
