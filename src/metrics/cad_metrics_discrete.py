import torch
import torch.nn as nn
from torchmetrics import Metric, MetricCollection


# -----------------------------------------------------------
# CE Per Class
# -----------------------------------------------------------
class CEPerClass(Metric):
    full_state_update = False
    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.softmax = nn.Softmax(dim=-1)
        self.bce = nn.BCELoss(reduction='sum')

    def update(self, preds, target):
        preds = preds.reshape(-1, preds.shape[-1])
        target = target.reshape(-1, target.shape[-1])
        mask = (target != 0).any(dim=-1)
        if mask.sum() == 0:
            return

        prob = self.softmax(preds)[mask][:, self.class_id]
        tgt = target[mask][:, self.class_id]

        self.total_ce += self.bce(prob, tgt)
        self.total_samples += prob.numel()

    def compute(self):
        if self.total_samples == 0:
            return torch.tensor(0.0, device=self.total_ce.device)
        return self.total_ce / self.total_samples


# -----------------------------------------------------------
# Node Type CE: 3 classes
# -----------------------------------------------------------
class NodeTypeMetricsCE(MetricCollection):
    def __init__(self, num_classes=3):
        metrics = {
            f"type_class_{i}": CEPerClass(i)
            for i in range(num_classes)
        }
        super().__init__(metrics)




# -----------------------------------------------------------
# Edge CE: 2 classes (no-edge, edge)
# -----------------------------------------------------------
class EdgeMetricsCE(MetricCollection):
    def __init__(self):
        metrics = {
            "edge_no_edge": CEPerClass(0),
            "edge_edge": CEPerClass(1),
        }
        super().__init__(metrics)


# -----------------------------------------------------------
# Geometric MSE: 10 dims (regression)
# -----------------------------------------------------------
class GeomMSE(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("sum_sq", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, pred_geom, true_geom, node_mask):
        """
        pred_geom: (bs, n, 10)
        true_geom: (bs, n, 10)
        node_mask: (bs, n)
        """
        mask = node_mask.unsqueeze(-1)  # (bs, n, 1)
        diff = (pred_geom - true_geom) * mask
        self.sum_sq += (diff ** 2).sum()
        self.count += mask.sum()

    def compute(self):
        if self.count == 0:
            return torch.tensor(0.0, device=self.sum_sq.device)
        return self.sum_sq / self.count



# -----------------------------------------------------------
# The final hybrid CAD metrics class
# -----------------------------------------------------------
class TrainCADMetricsHybrid(nn.Module):
    """
    CAD Metrics:
      - CE on type (3 dims)
      - MSE on geometry (10 dims)
      - CE on edge (2 dims)
    """
    def __init__(self):
        super().__init__()
        self.node_type_metrics = NodeTypeMetricsCE()
        self.edge_metrics = EdgeMetricsCE()
        self.geom_metrics = GeomMSE()

    def forward(self, pred_X, pred_E, true_X, true_E, node_mask, log=False):
        """
        pred_X: logits (bs, n, 13)
        true_X: one-hot (bs, n, 13)
        pred_E: logits (bs, n, n, 2)
        true_E: one-hot (bs, n, n, 2)
        node_mask: (bs, n)
        """

        # ---- CE on type (first 3 dims) ----
        self.node_type_metrics(pred_X[..., :3], true_X[..., :3])

        # ---- MSE on geometry (last 10 dims) ----
        self.geom_metrics(pred_X[..., 3:], true_X[..., 3:], node_mask)

        # ---- CE on edges ----
        self.edge_metrics(pred_E, true_E)

        # wandb logging
        if log:
            to_log = {}

            # node type CE
            for key, val in self.node_type_metrics.compute().items():
                to_log[f"train/type_ce/{key}"] = float(val)

            # edge CE
            for key, val in self.edge_metrics.compute().items():
                to_log[f"train/edge_ce/{key}"] = float(val)

            # geometry MSE
            to_log["train/geom_mse"] = float(self.geom_metrics.compute())

            import wandb
            if wandb.run:
                wandb.log(to_log, commit=False)

    def reset(self):
        self.node_type_metrics.reset()
        self.edge_metrics.reset()
        self.geom_metrics.reset()

    def log_epoch_metrics(self):
        node_type = self.node_type_metrics.compute()
        edge_type = self.edge_metrics.compute()
        geom_mse = self.geom_metrics.compute()

        to_log = {}

        for key, val in node_type.items():
            to_log[f"epoch/type_ce/{key}"] = float(val)

        for key, val in edge_type.items():
            to_log[f"epoch/edge_ce/{key}"] = float(val)

        to_log["epoch/geom_mse"] = float(geom_mse)

        import wandb
        if wandb.run:
            wandb.log(to_log, commit=False)

        # ---- Return two values only (compatibility with DiGress) ----
        epoch_node_metrics = {
            **{k: float(v) for k, v in node_type.items()},   # CE
            "geom_mse": float(geom_mse),                     # include geom in node dict
        }

        epoch_edge_metrics = {
            k: float(v) for k, v in edge_type.items()
        }

        return epoch_node_metrics, epoch_edge_metrics
