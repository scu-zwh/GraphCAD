import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics import Metric, MeanSquaredError, MetricCollection
import time
import wandb
from src.metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchMSE, SumExceptBatchKL, CrossEntropyMetric, \
    ProbabilityMetric, NLL


class NodeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class EdgeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class TrainLoss(nn.Module):
    def __init__(self):
        super(TrainLoss, self).__init__()
        self.train_node_mse = NodeMSE()
        self.train_edge_mse = EdgeMSE()
        self.train_y_mse = MeanSquaredError()

    def forward(self, masked_pred_epsX, masked_pred_epsE, pred_y, true_epsX, true_epsE, true_y, log: bool):
        mse_X = self.train_node_mse(masked_pred_epsX, true_epsX) if true_epsX.numel() > 0 else 0.0
        mse_E = self.train_edge_mse(masked_pred_epsE, true_epsE) if true_epsE.numel() > 0 else 0.0
        mse_y = self.train_y_mse(pred_y, true_y) if true_y.numel() > 0 else 0.0
        mse = mse_X + mse_E + mse_y

        if log:
            to_log = {'train_loss/batch_mse': mse.detach(),
                      'train_loss/node_MSE': self.train_node_mse.compute(),
                      'train_loss/edge_MSE': self.train_edge_mse.compute(),
                      'train_loss/y_mse': self.train_y_mse.compute()}
            if wandb.run:
                wandb.log(to_log, commit=True)

        return mse

    def reset(self):
        for metric in (self.train_node_mse, self.train_edge_mse, self.train_y_mse):
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_mse = self.train_node_mse.compute() if self.train_node_mse.total > 0 else -1
        epoch_edge_mse = self.train_edge_mse.compute() if self.train_edge_mse.total > 0 else -1
        epoch_y_mse = self.train_y_mse.compute() if self.train_y_mse.total > 0 else -1

        to_log = {"train_epoch/epoch_X_mse": epoch_node_mse,
                  "train_epoch/epoch_E_mse": epoch_edge_mse,
                  "train_epoch/epoch_y_mse": epoch_y_mse}
        if wandb.run:
            wandb.log(to_log)
        return to_log



class TrainLossDiscrete(nn.Module):
    """
    Hybrid CAD loss:
      - X_type (前 3 维): CrossEntropy
      - X_geom (后 10 维): MSE
      - E (2 维 one-hot): CrossEntropy
    """
    def __init__(self, lambda_train, dataset_infos):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        
        # added
        self.dataset_infos = dataset_infos
        self.mse = nn.MSELoss(reduction='mean')
        
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_train = lambda_train

    def forward(self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y, log: bool):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean. """
        bs, n, dx = true_X.size()
        device = true_X.device
        geom_dim = dx - 3

        # -------------------------
        # 2. Split X into type + geom
        # -------------------------
        pred_type_logits = masked_pred_X[..., :3]                      # (bs,n,3)
        true_X_type = true_X[..., :3]                             # (bs,n,3)

        pred_geom = masked_pred_X[..., 3:]                             # (bs,n,10)
        true_geom = true_X[..., 3:]                             # (bs,n,10)
        # 在这里添加归一化，使用self.dataset_infos.geom_mean和self.dataset_infos.geom_std
        # 改成传给网络归一化的数据
        # true_geom = (true_geom - self.dataset_infos.geom_mean.to(device)) / self.dataset_infos.geom_std.to(device)
        # pred_geom = (pred_geom - self.dataset_infos.geom_mean.to(device)) / self.dataset_infos.geom_std.to(device)
        
        # CE for type
        true_X_type = torch.reshape(true_X_type, (-1, true_X_type.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X_type = torch.reshape(pred_type_logits, (-1, pred_type_logits.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X_type = (true_X_type != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X_type = true_X_type[mask_X_type, :]
        flat_pred_X_type = masked_pred_X_type[mask_X_type, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_X = self.node_loss(flat_pred_X_type, flat_true_X_type) if true_X_type.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        loss_y = self.y_loss(pred_y, true_y) if true_y.numel() > 0 else 0.0
        
        # -------------------------
        # MSE loss for geom
        # -------------------------
        pred_geom = pred_geom.reshape(-1, geom_dim)[mask_X_type]
        true_geom = true_geom.reshape(-1, geom_dim)[mask_X_type]

        loss_geom = self.mse(pred_geom, true_geom)

        if log:
            to_log = {"train_loss/batch_CE": (loss_X + loss_E + loss_y + loss_geom).detach(),
                      "train_loss/X_CE": self.node_loss.compute() if true_X.numel() > 0 else -1,
                      "train_loss/geom_MSE": loss_geom.detach() if true_X.numel() > 0 else -1,
                      "train_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1,
                      "train_loss/y_CE": self.y_loss.compute() if true_y.numel() > 0 else -1}
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y + self.lambda_train[2] * loss_geom

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        epoch_y_loss = self.train_y_loss.compute() if self.y_loss.total_samples > 0 else -1

        to_log = {"train_epoch/x_CE": epoch_node_loss,
                  "train_epoch/E_CE": epoch_edge_loss,
                  "train_epoch/y_CE": epoch_y_loss}
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log



