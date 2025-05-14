import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops.misc import FrozenBatchNorm2d


class MlpPolicy(nn.Module):
    """MLP policy with ResNet backbone."""

    def __init__(
        self,
        state_dim,
        action_dim,
        num_images,
        horizon,
        n_obs_steps,
        n_action_steps,
        hidden_dim_list,
        state_feature_dim,
    ):
        super().__init__()

        # Setup Variable
        self.horizon = horizon if n_obs_steps > 1 or n_action_steps > 1 else 1
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps

        # Instantiate state feature extractor
        self.state_feature_extractor = nn.Sequential(
            nn.Linear(state_dim * self.horizon, state_feature_dim),
            # nn.BatchNorm1d(state_feature_dim),
            nn.ReLU(),
        )

        # Instantiate image feature extractor
        resnet_model = resnet18(
            weights=ResNet18_Weights.DEFAULT, norm_layer=FrozenBatchNorm2d
        )
        self.image_feature_extractor = nn.Sequential(
            *list(resnet_model.children())[:-1]
        )  # Remove last layer
        image_feature_dim = resnet_model.fc.in_features

        # Instantiate linear layers
        combined_feature_dim = (
            state_feature_dim + num_images * image_feature_dim * self.horizon
        )
        linear_dim_list = (
            [combined_feature_dim] + hidden_dim_list + [action_dim * self.horizon]
        )
        linear_layers = []
        for linear_idx in range(len(linear_dim_list) - 1):
            input_dim = linear_dim_list[linear_idx]
            output_dim = linear_dim_list[linear_idx + 1]
            linear_layers += [nn.Linear(input_dim, output_dim)]
            if linear_idx < len(linear_dim_list) - 2:
                linear_layers += [
                    # nn.BatchNorm1d(output_dim),
                    nn.ReLU(),
                ]
        self.linear_layer_seq = nn.Sequential(*linear_layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, states, whole_images):
        if self.n_obs_steps > 1:
            batch_size, num_obs_steps, num_images, C, H, W = whole_images.shape
            _, num_state_obs_steps, state_hdim = states.shape
            if num_state_obs_steps < self.horizon and not self.training:
                pad_len = self.horizon - num_state_obs_steps
                states = torch.cat(
                    [states]
                    + [
                        states.clone()[:, -1].reshape(batch_size, 1, state_hdim)
                        for _ in range(pad_len)
                    ],
                    dim=1,
                ).to(states.device)
            if num_obs_steps < self.horizon and not self.training:
                pad_len = self.horizon - num_obs_steps
                whole_images = torch.cat(
                    [whole_images]
                    + [
                        whole_images.clone()[:, -1].reshape(
                            batch_size, 1, num_images, C, H, W
                        )
                        for _ in range(pad_len)
                    ],
                    dim=1,
                ).to(whole_images.device)
            # Extract state, images from states, whole_images
            state = states.reshape(batch_size, -1)
            images = whole_images.reshape(batch_size, -1, C, H, W)
        else:
            batch_size, num_images, C, H, W = whole_images.shape
            state = states
            images = whole_images

        # Extract state feature
        state_feature = self.state_feature_extractor(
            state
        )  # (batch_size, state_feature_dim)

        # Extract image feature
        image_features = []
        for i in range(self.horizon):
            image_feature = self.image_feature_extractor(
                images[:, i]
            )  # (batch_size, image_feature_dim, 1, 1)
            image_feature = image_feature.view(
                batch_size, -1
            )  # (batch_size, image_feature_dim)
            image_features.append(image_feature)
        image_features = torch.cat(
            image_features, dim=1
        )  # (batch_size, num_images * horizon * image_feature_dim)

        # Apply linear layers
        combined_feature = torch.cat(
            [state_feature, image_features], dim=1
        )  # (batch_size, combined_feature_dim)
        action_feature = self.linear_layer_seq(
            combined_feature
        )  # (batch_size, action_dim * horizon)
        if self.n_action_steps > 1:
            action = action_feature.reshape(
                batch_size, self.horizon, -1
            )  # (batch_size, horizon, action_dim)
        else:
            action = action_feature  # (batch_size, action_dim)

        return action
