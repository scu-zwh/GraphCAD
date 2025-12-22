from torchmetrics import MeanSquaredError, MeanAbsoluteError

### packages for visualization
import torch
from torchmetrics import Metric, MetricCollection
from torch_geometric.data import Data
from torch import Tensor
import wandb
import torch.nn as nn

from torchmetrics import Metric, MetricCollection
import torch
import torch.nn as nn

from analysis.cad_utils import compute_cad_metrics
from analysis.spectre_utils import Comm20SamplingMetrics

class NodeMetrics(MetricCollection):
    """
    CAD node metrics for diffusion noise:
    - dim 0..type_dim-1: type-related eps
    - dim type_dim..end: geometry-related eps
    """
    def __init__(self, type_dim: int = 3):
        self.type_dim = type_dim
        metrics = {
            "node_mse_all": NodeMSEAll(type_dim=type_dim),
            "node_mse_type": NodeMSEType(type_dim=type_dim),
            "node_mse_geom": NodeMSEGeom(type_dim=type_dim),
        }
        super().__init__(metrics)


class NodeMSEAll(Metric):
    """ 对整条 node feature 的 eps 做 MSE """
    def __init__(self, type_dim: int = 3):
        super().__init__()
        self.type_dim = type_dim
        self.add_state("sum_sq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_epsX: torch.Tensor, true_epsX: torch.Tensor):
        # pred/true: (bs, n, d)
        diff = pred_epsX - true_epsX           # (bs, n, d)
        mse_per_node = (diff ** 2).mean(dim=-1)  # (bs, n)
        self.sum_sq += mse_per_node.sum()
        self.count += mse_per_node.numel()

    def compute(self):
        if self.count == 0:
            return torch.tensor(0.0, device=self.sum_sq.device)
        return self.sum_sq / self.count


class NodeMSEType(Metric):
    """ 只对前 type_dim 维（原来类型对应的那块 eps）做 MSE """
    def __init__(self, type_dim: int = 3):
        super().__init__()
        self.type_dim = type_dim
        self.add_state("sum_sq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_epsX: torch.Tensor, true_epsX: torch.Tensor):
        pred_type = pred_epsX[..., :self.type_dim]   # (bs, n, type_dim)
        true_type = true_epsX[..., :self.type_dim]
        diff = pred_type - true_type
        mse_per_node = (diff ** 2).mean(dim=-1)      # (bs, n)
        self.sum_sq += mse_per_node.sum()
        self.count += mse_per_node.numel()

    def compute(self):
        if self.count == 0:
            return torch.tensor(0.0, device=self.sum_sq.device)
        return self.sum_sq / self.count


class NodeMSEGeom(Metric):
    """ 只对几何部分（后面的 10 维 eps）做 MSE """
    def __init__(self, type_dim: int = 3):
        super().__init__()
        self.type_dim = type_dim
        self.add_state("sum_sq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_epsX: torch.Tensor, true_epsX: torch.Tensor):
        pred_geom = pred_epsX[..., self.type_dim:]   # (bs, n, 10)
        true_geom = true_epsX[..., self.type_dim:]
        diff = pred_geom - true_geom
        mse_per_node = (diff ** 2).mean(dim=-1)      # (bs, n)
        self.sum_sq += mse_per_node.sum()
        self.count += mse_per_node.numel()

    def compute(self):
        if self.count == 0:
            return torch.tensor(0.0, device=self.sum_sq.device)
        return self.sum_sq / self.count


class EdgeMetrics(Metric):
    """Compute overall MSE for edge diffusion noise epsE."""
    def __init__(self):
        super().__init__()
        self.add_state("sum_sq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_epsE: torch.Tensor, true_epsE: torch.Tensor):
        """
        pred_epsE: (bs, n, n, dE)
        true_epsE: (bs, n, n, dE)
        """
        diff = pred_epsE - true_epsE                      # (bs, n, n, dE)
        mse_per_edge = (diff ** 2).mean(dim=-1)           # (bs, n, n)

        # 全部纳入统计（如果你想排除对角线，也可加入 mask）
        self.sum_sq += mse_per_edge.sum()
        self.count += mse_per_edge.numel()

    def compute(self):
        if self.count == 0:
            mse = torch.tensor(0.0, device=self.sum_sq.device)
        else:
            mse = self.sum_sq / self.count
        return {"edge_mse": mse}

    def reset(self):
        self.sum_sq = torch.tensor(0.0, device=self.sum_sq.device)
        self.count = torch.tensor(0, device=self.count.device)

class TrainCADMetrics(nn.Module):
    """Metrics for CAD node features (type classification + geometry regression)."""
    def __init__(self):
        super().__init__()
        self.train_node_metrics = NodeMetrics()
        self.train_edge_metrics = EdgeMetrics()

    def forward(self, masked_pred_epsX, masked_pred_epsE, pred_y, true_epsX, true_epsE, true_y, log: bool):
        self.train_node_metrics(masked_pred_epsX, true_epsX)
        self.train_edge_metrics(masked_pred_epsE, true_epsE)
        
        if log:
            to_log = {}
            for key, val in self.train_node_metrics.compute().items():
                to_log['train/' + key] = val.item()
            for key, val in self.train_edge_metrics.compute().items():
                to_log['train/' + key] = val.item()
            if wandb.run:
                wandb.log(to_log, commit=False)

    def reset(self):
        for metric in [self.train_node_metrics, self.train_edge_metrics]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_metrics = self.train_node_metrics.compute()
        epoch_edge_metrics = self.train_edge_metrics.compute()

        to_log = {}
        for key, val in epoch_node_metrics.items():
            to_log['train_epoch/epoch' + key] = val.item()
        for key, val in epoch_edge_metrics.items():
            to_log['train_epoch/epoch' + key] = val.item()

        if wandb.run:
            wandb.log(to_log, commit=False)

        for key, val in epoch_node_metrics.items():
            epoch_node_metrics[key] = f"{val.item() :.3f}"
        for key, val in epoch_edge_metrics.items():
            epoch_edge_metrics[key] = f"{val.item() :.3f}"

        return epoch_node_metrics, epoch_edge_metrics

#############################################
#       Helper Metrics
#############################################

class HistogramMAE(Metric):
    """Compare generated histograms with target histograms."""
    def __init__(self, target_hist):
        super().__init__()
        self.register_buffer("target", target_hist)
        self.add_state("generated", default=torch.zeros_like(target_hist), dist_reduce_fx="mean")

    def update(self, generated_hist):
        device = self.generated.device
        self.generated = torch.as_tensor(generated_hist, device=device)


    def compute(self):
        return torch.mean(torch.abs(self.generated - self.target))

    def reset(self):
        self.generated.zero_()


#############################################
#       Main CAD Sampling Metrics
#############################################

class SamplingCADMetrics(nn.Module):
    def __init__(self, datamodule, dataset_infos, train_graph_signatures):
        super().__init__()
        self.dataset_infos = dataset_infos
        self.train_graph_signatures = train_graph_signatures
        
        
        di = dataset_infos

        # 1. Node count distribution
        n_target = di.n_nodes.float()
        n_target = n_target / n_target.sum()
        self.register_buffer("target_n_dist", n_target)
        self.n_mae = HistogramMAE(n_target)

        # 2. Node type distribution (3 types)
        node_target = di.node_types.float()
        node_target = node_target / node_target.sum()
        self.register_buffer("target_node_type_dist", node_target)
        self.node_type_mae = HistogramMAE(node_target)

        # 3. Geometry mean & std
        self.register_buffer("geom_target_mean", di.geom_mean)
        self.register_buffer("geom_target_std", di.geom_std)

        # 4. Edge density distribution
        edge_target = di.edge_density.float()
        self.register_buffer("target_edge_density", edge_target)
        self.edge_density_mae = HistogramMAE(edge_target)

        # ⭐ 新增：COMM20 相关指标
        self.comm20_metrics = Comm20SamplingMetrics(datamodule)
        
    def pyg_to_adj_list(self, pyg_graphs):
        adj_list = []
        for g in pyg_graphs:
            n = g.x.size(0)
            A = torch.zeros((n, n), dtype=torch.float32)

            src, dst = g.edge_index
            A[src, dst] = 1
            A[dst, src] = 1  # 无向图

            adj_list.append(A)
        return adj_list


    #############################################
    #           Forward: compute all metrics
    #############################################
    def forward(self, generated_graphs,  name, current_epoch, val_counter, test=False):
        """
        generated_graphs: list of [X, E]
            X: (n, 13)
            E: (n, n, 2)
        Each Data.x contains (3 onehot + 10 geom)
        """
        
        # ------------------------------
        # Convert to PyG Data objects
        # ------------------------------
        pyg_graphs = []
        for item in generated_graphs:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(f"Unexpected generated graph format: {item}")

            X, E = (t.to(self.target_node_type_dist.device) for t in item)  # X=(n, 13), E=(n, n, 2)

            # --------------------------------
            # 1. Build edge_index and edge_attr
            # --------------------------------
            # E[..., 1] is the “has-edge” channel (0/1)
            edge_type = E.argmax(dim=-1)  # shape (n,n), 0=no-edge, 1=edge
            src, dst = torch.nonzero(edge_type == 1, as_tuple=True)
            edge_index = torch.stack([src, dst], dim=0)
            edge_attr = E[src, dst]   # still 2-dim one-hot

            # --------------------------------
            # 2. Convert to PyG Data
            # --------------------------------
            data = Data(
                x=X, 
                edge_index=edge_index, 
                edge_attr=edge_attr
            )
            pyg_graphs.append(data)

        # Now replace original generated_graphs
        generated_graphs = pyg_graphs


        #############################
        # 1. Node count distribution
        #############################
        n_list = [g.x.shape[0] for g in generated_graphs]
        max_n = self.target_n_dist.shape[0]
        hist = torch.zeros(max_n, device=self.target_n_dist.device)
        for n in n_list:
            if n < max_n:
                hist[n] += 1
        hist = hist / hist.sum()
        self.n_mae(hist)

        #############################
        # 2. Node type distribution
        #############################
        node_type_counts = torch.zeros(3, device=self.target_node_type_dist.device)
        total_nodes = 0
        for g in generated_graphs:
            types = g.x[:, :3].argmax(dim=-1)
            for t in range(3):
                node_type_counts[t] += (types == t).sum()
            total_nodes += g.x.shape[0]
        node_type_dist = node_type_counts / (total_nodes + 1e-6)
        self.node_type_mae(node_type_dist)

        #############################
        # 3. Geometry feature stats
        #############################
        geom_vals = torch.cat([g.x[:, 3:] for g in generated_graphs], dim=0)
        geom_mean = geom_vals.mean(dim=0)
        geom_std = geom_vals.std(dim=0)

        geom_mean_diff = geom_mean - self.geom_target_mean
        geom_std_diff = geom_std - self.geom_target_std

        #############################
        # 4. Edge density
        #############################
        densities = []
        for g in generated_graphs:
            n = g.x.shape[0]
            if n <= 1:
                continue
            e = g.edge_index
            num_edges = e.shape[1] / 2   # undirected
            densities.append(num_edges / (n*n))
        gen_edge_density = torch.tensor(densities).mean()
        edge_density_diff = gen_edge_density - self.target_edge_density

        self.edge_density_mae(torch.tensor([gen_edge_density]))

        #############################
        # 5. Uniqueness & Valid
        #############################
        
        vun = compute_cad_metrics(generated_graphs, self.train_graph_signatures, self.dataset_infos)
        validity   = vun["validity"]
        uniqueness = vun["uniqueness"]
        novelty    = vun["novelty"]
        cc_mean   = vun["connected_components"]['mean']
        
        
        ############################################
        # 6. Comm20 graph structure metrics
        ############################################
        # 将 CAD PyG 图转换为 adjacency matrices
        adj_list = self.pyg_to_adj_list(generated_graphs)

        # Comm20SamplingMetrics 接受的是:
        #    [(node_types, edge_matrix), ...]
        generated_formatted = []
        for i, A in enumerate(adj_list):
            x = generated_graphs[i].x
            node_types = x[:, :3].argmax(dim=-1)  # node type id
            generated_formatted.append((node_types, A))

        # 运行 COMM20 metrics
        comm20_results = self.comm20_metrics(
            generated_formatted,
            name=name,
            current_epoch=current_epoch,
            val_counter=val_counter,
            local_rank=0,
            test=test
        )  

        #############################
        # Prepare output
        #############################
        metrics = {
            "cad_metrics/n_mae": self.n_mae.compute().item(),
            "cad_metrics/type_mae": self.node_type_mae.compute().item(),
            "cad_metrics/geom_mean_mae": geom_mean_diff.abs().mean().item(),
            "cad_metrics/geom_std_mae": geom_std_diff.abs().mean().item(),
            "cad_metrics/edge_density_mae": self.edge_density_mae.compute().item(),
            "cad_metrics/validity": validity,
            "cad_metrics/uniqueness": uniqueness,
            "cad_metrics/novelty": novelty,
            "cad_metrics/cc_mean": cc_mean,
            "cad_metrics/degree": comm20_results.get("degree", 0),
            "cad_metrics/clustering": comm20_results.get("clustering", 0),
            "cad_metrics/orbit": comm20_results.get("orbit", 0)   
        }

        print(f"[{name}][Epoch {current_epoch}] CAD Sampling Metrics at sample #{val_counter}:")
        print(metrics)
        
        if wandb.run:
            wandb.log(metrics, commit=False)

        return metrics

    def reset(self):
        self.n_mae.reset()
        self.node_type_mae.reset()
        self.edge_density_mae.reset()

if __name__ == '__main__':
    from torchmetrics.utilities import check_forward_full_state_property
