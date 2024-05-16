from typing import Tuple

import torch
import torch.nn as nn
from eipl.utils import LossScheduler, tensor2numpy

from torch import nn, optim, Tensor
from torch.utils.data import DataLoader

from eipl.model import SARNN


float_2_t = Tuple[float, float]
float_3_t = Tuple[float, float, float]


class fullBPTTtrainer:
    """
    Helper class to train recurrent neural networks with numpy sequences

    Args:
        traindata (np.array): list of np.array. First diemension should be time steps
        model (torch.nn.Module): rnn model
        optimizer (torch.optim): optimizer
        input_param (float): input parameter of sequential generation. 1.0 means open mode.
    """

    def __init__(
        self,
        model: SARNN,
        optimizer: optim.Optimizer,
        loss_weights: float_3_t = (1.0, 1.0, 1.0),
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.optimizer = optimizer
        self.loss_weights = loss_weights
        self.scheduler = LossScheduler(decay_end=1000, curve_name="s")
        self.model = model.to(self.device)

    def save(
        self,
        epoch: int,
        loss: float_2_t,
        savename: str,
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                #'optimizer_state_dict': self.optimizer.state_dict(),
                "train_loss": loss[0],
                "test_loss": loss[1],
            },
            savename,
        )

    def process_epoch(
        self,
        dataloader: DataLoader,
        training: bool = True,
    ) -> float:
        x_img: Tensor
        x_joint: Tensor
        y_img: Tensor
        y_joint: Tensor
        mask: Tensor

        criterion = nn.MSELoss(reduction="none")

        if not training:
            self.model.eval()
        else:
            self.model.train()

        total_loss = 0.0
        for n_batch, ((x_img, x_joint), (y_img, y_joint), mask) in enumerate(
            dataloader
        ):
            x_img = x_img.to(self.device)
            x_joint = x_joint.to(self.device)
            y_img = y_img.to(self.device)
            y_joint = y_joint.to(self.device)
            mask = mask.to(self.device)

            batch_size, timesteps = x_img.size()[:2]
            state = None

            yi_hat = torch.empty_like(y_img[:, 1:])
            yv_hat = torch.empty_like(y_joint[:, 1:])
            enc_pts = torch.empty(
                (batch_size, timesteps - 1, 2 * self.model.k_dim),
                dtype=x_img.dtype,
                device=x_img.device,
            )
            dec_pts = enc_pts.clone()

            self.optimizer.zero_grad(set_to_none=True)
            for t in range(x_img.shape[1] - 1):
                yi_hat_t, yv_hat_t, enc_ij, dec_ij, state = self.model(
                    x_img[:, t], x_joint[:, t], state
                )
                yi_hat[:, t] = yi_hat_t
                yv_hat[:, t] = yv_hat_t
                enc_pts[:, t] = enc_ij
                dec_pts[:, t] = dec_ij

            # [TODO] mask, MSELoss(reduction="none")

            img_loss = torch.mean(
                criterion(yi_hat, y_img[:, 1:]) * self.loss_weights[0],
                dim=(2, 3, 4),
            )
            joint_loss = torch.mean(
                criterion(yv_hat, y_joint[:, 1:]) * self.loss_weights[1],
                dim=2,
            )
            loss = img_loss + joint_loss
            loss = torch.sum(loss * mask[:, 1:]) / torch.sum(mask[:, 1:])
            # Gradually change the loss value using the LossScheluder class.
            pt_loss = torch.mean(
                criterion(dec_pts[:, :-1], enc_pts[:, 1:]),
                dim=2,
            ) * self.scheduler(self.loss_weights[2])
            loss += torch.sum(pt_loss * mask[:, 1:-1]) / torch.sum(
                mask[:, 1:-1]
            )
            total_loss += tensor2numpy(loss)

            if training:
                loss.backward()
                self.optimizer.step()

        return total_loss / (n_batch + 1)
