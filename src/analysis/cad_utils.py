import torch
import wandb
import numpy as np
from torch_geometric.data import Data

class BasicCADMetrics(object):

    def __init__(self, dataset_infos, train_graph_signatures=None):
        """
        type_decoder = ['LINE','ARC','CIRCLE']
        train_graph_signatures: set(), 用于 novelty 检查
        """
        self.type_decoder = dataset_infos.atom_decoder
        self.dataset_infos = dataset_infos
        self.train_graph_signatures = train_graph_signatures

    # ------------------------------------------------------------
    # Geometry Checks
    # ------------------------------------------------------------

    def check_line(self, P0, P1, tol=1e-5):
        return torch.norm(P0 - P1) > tol

    def check_arc(self, P0, P1, C, r, tol=1e-3):
        if r <= 0:
            return False
        d0 = torch.norm(P0 - C)
        d1 = torch.norm(P1 - C)
        return (abs(d0 - r) < tol) and (abs(d1 - r) < tol)

    # ------------------------------------------------------------
    # Topology Checks: adjacency consistency
    # ------------------------------------------------------------

    def check_adjacency(self, X, edge_index, eps=2e-2):
        """
        在 PyG Data 图中检查拓扑一致性。
        X: (n,12)
        edge_index: (2,m)
        """
        n = X.size(0)
        P0 = X[:, 3:6]
        P1 = X[:, 6:9]

        for u, v in edge_index.t():
            u, v = int(u), int(v)
            # 棱必须几何连接：四种端点组合其一距离很小
            d = torch.tensor([
                torch.norm(P0[u] - P0[v]),
                torch.norm(P0[u] - P1[v]),
                torch.norm(P1[u] - P0[v]),
                torch.norm(P1[u] - P1[v])
            ])
            if torch.min(d) > eps:
                return False
        return True

    # ------------------------------------------------------------
    # Graph signature for uniqueness
    # ------------------------------------------------------------

    def make_signature(self, X):
        """
        返回 CAD 图的几何 signature，用于 uniqueness / novelty
        """
        types = X[:, :3].argmax(dim=-1).cpu().numpy()
        P0 = X[:, 3:6].cpu().numpy()
        P1 = X[:, 6:9].cpu().numpy()
        P2 = X[:, 9:12].cpu().numpy()

        # 用长度 + 类型 + 半径构建 signature
        lengths = np.linalg.norm(P1 - P0, axis=1)
        r = np.linalg.norm(P0 - P2, axis=1)  # 对于 ARC，半径；对于其他类型，点到点距离作为“伪半径”

        sig = list(zip(types.tolist(), lengths.tolist(), r.tolist()))
        sig = sorted(sig)
        return tuple(sig)

    # ------------------------------------------------------------
    # Main validity function
    # ------------------------------------------------------------

    def check_geometry(self, X):
        """
        X: (n,13)
        """
        type_id = X[:, :3].argmax(dim=-1)

        for i, t in enumerate(type_id):
            P0 = X[i, 3:6]
            P1 = X[i, 6:9]

            if t == 0:  # LINE
                if not self.check_line(P0, P1):
                    return False

            else:  # ARC / CIRCLE
                C = X[i, 9:12]
                r = np.linalg.norm(P0 - C, axis=1)
                if not self.check_arc(P0, P1, C, r):
                    return False

        return True

    # ------------------------------------------------------------
    # Connectivity (PyG)
    # ------------------------------------------------------------

    def connected_components(self, data: Data):
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        edge_list = data.edge_index.t().cpu().numpy().tolist()
        G.add_edges_from(edge_list)
        return nx.number_connected_components(G)

    # ------------------------------------------------------------
    # Evaluate a list of PyG CAD graphs
    # ------------------------------------------------------------

    def evaluate(self, pyg_graphs):
        valid_list = []
        comp_list = []
        signatures = []

        for g in pyg_graphs:
            X = g.x
            edge_index = g.edge_index

            ok_geom = self.check_geometry(X)
            ok_topo = self.check_adjacency(X, edge_index)
            is_valid = ok_geom and ok_topo

            valid_list.append(is_valid)
            comp_list.append(self.connected_components(g))

            if is_valid:
                signatures.append(self.make_signature(X))

        # Validity
        validity = sum(valid_list) / len(valid_list)

        # Uniqueness
        unique_set = set(signatures)
        uniqueness = len(unique_set) / max(1, len(signatures))

        # Novelty（可选）
        if self.train_graph_signatures is not None:
            novel = [s for s in unique_set if s not in self.train_graph_signatures]
            novelty = len(novel) / max(1, len(unique_set))
        else:
            novelty = -1

        return {
            "validity": validity,
            "connected_components": {
                "min": min(comp_list),
                "max": max(comp_list),
                "mean": float(np.mean(comp_list)),
            },
            "uniqueness": uniqueness,
            "novelty": novelty,
        }
        
        
def compute_cad_metrics(graph_list, train_graph_signatures, dataset_infos):
    """
    graph_list: list of PyG Data objects (x, edge_index, edge_attr)
    """

    cad_metrics = BasicCADMetrics(
        dataset_infos=dataset_infos,              # ["SPLINE","ARC","LINE"]
        train_graph_signatures=train_graph_signatures        # 用于 novelty
    )

    # Evaluate full metrics
    result = cad_metrics.evaluate(graph_list)

    # 拆分结果
    validity      = result["validity"]
    uniqueness    = result["uniqueness"]
    novelty       = result["novelty"]
    cc_info       = result["connected_components"]

    # wandb logging
    if wandb.run:
        wandb.log({
            "CAD/Validity": validity,
            "CAD/Uniqueness": uniqueness,
            "CAD/Novelty": novelty,
            "CAD/CC_min": cc_info["min"],
            "CAD/CC_max": cc_info["max"],
            "CAD/CC_mean": cc_info["mean"]
        })

    return result


