import torch
import torch.nn as nn
from eipl.utils import LossScheduler, tensor2numpy
from eipl.tutorials.airec.sarnn.libs.fullBPTT import fullBPTTtrainer

class fullBPTTtrainerWithMask(fullBPTTtrainer):
    def process_epoch(self, data, training=True):
        if not training:
            self.model.eval()
        else:
            self.model.train()

        total_loss = 0.0
        for n_batch, ((x_img, x_joint), (y_img, y_joint), mask) in enumerate(data):
            if "cpu" in self.device:
                x_img = x_img.to(self.device)
                y_img = y_img.to(self.device)
                x_joint = x_joint.to(self.device)
                y_joint = y_joint.to(self.device)
                mask = mask.to(self.device)

            state = None
            yi_list, yv_list = [], []
            dec_pts_list, enc_pts_list = [], []
            self.optimizer.zero_grad(set_to_none=True)
            for t in range(x_img.shape[1] - 1):
                _yi_hat, _yv_hat, enc_ij, dec_ij, state = self.model(
                    x_img[:, t], x_joint[:, t], state
                )
                yi_list.append(_yi_hat)
                yv_list.append(_yv_hat)
                enc_pts_list.append(enc_ij)
                dec_pts_list.append(dec_ij)

            yi_hat = torch.permute(torch.stack(yi_list), (1, 0, 2, 3, 4))
            yv_hat = torch.permute(torch.stack(yv_list), (1, 0, 2))

            criterion = nn.MSELoss(reduction="none")

            img_loss = torch.mean(
                criterion(yi_hat, y_img[:, 1:]),
                dim=(2, 3, 4))
            masked_img_loss = torch.sum(img_loss * mask[:, 1:]) / torch.sum(mask[:, 1:])

            joint_loss = torch.mean(
                criterion(yv_hat, y_joint[:, 1:]),
                dim=2)
            masked_joint_loss = torch.sum(joint_loss * mask[:, 1:]) / torch.sum(mask[:, 1:])

            pt_loss = torch.mean(
                criterion(torch.stack(dec_pts_list[:-1]), torch.stack(enc_pts_list[1:])) ,
                dim=2)
            masked_pt_loss = torch.sum(pt_loss * mask[:, 1:-1]) / torch.sum(mask[:, 1:-1])

            loss = self.loss_weights[0] * masked_img_loss + \
                self.loss_weights[1] * masked_joint_loss + \
                self.scheduler(self.loss_weights[2]) * masked_pt_loss
            total_loss += tensor2numpy(loss)

            if training:
                loss.backward()
                self.optimizer.step()

        return total_loss / (n_batch + 1)
