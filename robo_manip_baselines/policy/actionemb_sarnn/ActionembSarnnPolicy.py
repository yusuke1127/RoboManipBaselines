from collections.abc import Iterable
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
from eipl.layer import InverseSpatialSoftmax, SpatialSoftmax


class ActionembSarnnPolicy(nn.Module):
    """
    SARNN (Spatial attention recurrent neural network) policy.

    This implementation extends the original SARNN so that it can handle multiple images and additional data such as wrench. See the following repository for the original SARNN implementation: https://github.com/ogata-lab/eipl
    """

    def __init__(
        self,
        state_dim,
        num_images,
        act_inter_sizes: Union[int, Dict[str, int]],
        act_emb_sizes: Union[int, Dict[str, int]],
        image_size_list,
        num_attentions,
        lstm_hidden_dim,
    ):
        # Setup variables
        self.num_attentions = num_attentions

        kernel_size = 3
        softmax_temperature = 1e-4
        inv_softmax_heatmap_size = 0.1

        if isinstance(image_size_list[0], Iterable):
            assert num_images == len(image_size_list)
        else:
            image_size_list = [image_size_list] * num_images

        super().__init__()

        # Instantiate layers
        activation = nn.LeakyReLU(negative_slope=0.3)
        encoded_image_size_list = [
            (
                image_size[0] - 3 * (kernel_size - 1),
                image_size[1] - 3 * (kernel_size - 1),
            )
            for image_size in image_size_list
        ]

        self.attention_encoder_list = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(3, 16, kernel_size, 1, 0),
                activation,
                nn.Conv2d(16, 32, kernel_size, 1, 0),
                activation,
                nn.Conv2d(32, self.num_attentions, kernel_size, 1, 0),
                activation,
                SpatialSoftmax(
                    width=encoded_image_size[0],
                    height=encoded_image_size[1],
                    temperature=softmax_temperature,
                    normalized=True,
                ),
            )
            for encoded_image_size in encoded_image_size_list
        )

        self.image_encoder_list = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(3, 16, kernel_size, 1, 0),
                activation,
                nn.Conv2d(16, 32, kernel_size, 1, 0),
                activation,
                nn.Conv2d(32, self.num_attentions, kernel_size, 1, 0),
                activation,
            )
            for _ in range(num_images)
        )

        # Action Classifier
        self.classify_lstm = nn.LSTMCell(
            state_dim + num_images * self.num_attentions * 2, lstm_hidden_dim
        )

        if isinstance(act_inter_sizes, int):
            act_inter_sizes = {"action_0": act_inter_sizes}
        if isinstance(act_emb_sizes, int):
            act_emb_sizes = {"action_0": act_emb_sizes}
        
        self.action_inter = nn.ModuleDict()
        self.action_gate = nn.ModuleDict()
        self.action_embedding = nn.ModuleDict()

        total_act_size = 0
        for key in act_inter_sizes.keys():
            inter_size = act_inter_sizes[key]
            emb_size = act_emb_sizes[key]
            self.action_inter[key] = nn.Linear(lstm_hidden_dim, inter_size)
            self.action_gate[key] = nn.Linear(lstm_hidden_dim, 1)
            self.action_embedding[key] = nn.Linear(
                inter_size, emb_size, bias=False
            )
            total_act_size += inter_size

        self.predict_lstm = nn.LSTMCell(
            state_dim + num_images * self.num_attentions * 2 + total_act_size, lstm_hidden_dim
        )

        # Joint Decoder
        self.state_decoder = nn.Sequential(
            nn.Linear(lstm_hidden_dim, state_dim),
            activation,
        )

        self.attention_decoder_list = nn.ModuleList(
            nn.Sequential(
                nn.Linear(lstm_hidden_dim, self.num_attentions * 2),
                activation,
            )
            for _ in range(num_images)
        )

        self.inv_softmax_list = nn.ModuleList(
            InverseSpatialSoftmax(
                width=encoded_image_size[0],
                height=encoded_image_size[1],
                heatmap_size=inv_softmax_heatmap_size,
                normalized=True,
            )
            for encoded_image_size in encoded_image_size_list
        )

        self.image_decoder_list = nn.ModuleList(
            nn.Sequential(
                nn.ConvTranspose2d(self.num_attentions, 32, kernel_size, 1, 0),
                activation,
                nn.ConvTranspose2d(32, 16, kernel_size, 1, 0),
                activation,
                nn.ConvTranspose2d(16, 3, kernel_size, 1, 0),
                # activation,
            )
            for _ in range(num_images)
        )

        # Initialize weights
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.LSTMCell):
            nn.init.xavier_uniform_(m.weight_ih)
            nn.init.orthogonal_(m.weight_hh)
            nn.init.zeros_(m.bias_ih)
            nn.init.zeros_(m.bias_hh)
        elif (
            isinstance(m, nn.Conv2d)
            or isinstance(m, nn.ConvTranspose2d)
            or isinstance(m, nn.Linear)
        ):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, state, image_list, classify_state=None, predict_state=None):
        assert len(image_list) == len(self.image_encoder_list)

        # Encode images
        encoded_image_list = []
        for image, image_encoder in zip(image_list, self.image_encoder_list):
            encoded_image_list.append(image_encoder(image))

        # Calculate encoded attentions
        attention_list = []
        for image, attention_encoder in zip(image_list, self.attention_encoder_list):
            attention = attention_encoder(image)[0].reshape(-1, self.num_attentions * 2)
            attention_list.append(attention)

        # Forward LSTM <- concat with actionemb lstm
        classify_lstm_input = torch.cat([state, *attention_list], dim=-1)
        classify_lstm_output = self.classify_lstm(classify_lstm_input, classify_state)
        
        # action classification
        pred_rec_in: OrderedDict[str, Tensor] = OrderedDict(
            attention=attention_list,
            vector=state,
        )
        predicted_actionemb: OrderedDict[str, Tensor] = OrderedDict()
        for key in self.action_inter.keys():
            fc = self.action_inter[key]
            gate = self.action_gate[key]
            proj = self.action_embedding[key]

            inter = fc(classify_lstm_output[0])
            g = gate(classify_lstm_output[0])
            pred_rec_in[key] = inter * g

            embed = proj(inter)
            embed = embed / torch.norm(embed, dim=-1, keepdim=True)
            predicted_actionemb[key] = embed

        predict_lstm_input = torch.cat([classify_lstm_output, tuple(pred_rec_in.values())], dim=-1)
        predicted_lstm_output = self.predict_lstm(predict_lstm_input, predict_state)

        # Decode state
        predicted_state = self.state_decoder(predicted_lstm_output[0])

        # Decode attentions
        predicted_attention_list = []
        for attention_decoder in self.attention_decoder_list:
            predicted_attention = attention_decoder(predicted_lstm_output[0]).reshape(
                -1, self.num_attentions, 2
            )
            predicted_attention_list.append(predicted_attention)

        # Decode images
        predicted_image_list = []
        for encoded_image, predicted_attention, inv_softmax, image_decoder in zip(
            encoded_image_list,
            predicted_attention_list,
            self.inv_softmax_list,
            self.image_decoder_list,
        ):
            heatmap = inv_softmax(predicted_attention)
            predicted_image = image_decoder(torch.mul(heatmap, encoded_image))
            predicted_image_list.append(predicted_image)

        # Post-process
        attention_list = [
            attention.reshape(-1, self.num_attentions, 2)
            for attention in attention_list
        ]

        return (
            predicted_state,  # (batch_size, state_dim)
            predicted_image_list,  # (num_images, batch_size, 3, width, height)
            attention_list,  # (num_images, batch_size, num_attentions, 2)
            predicted_attention_list,  # (num_images, batch_size, num_attentions, 2)
            predicted_actionemb, 
            classify_lstm_output,
            predicted_lstm_output,
        )
