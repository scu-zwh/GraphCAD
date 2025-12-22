import os
import torch_geometric.utils
from omegaconf import OmegaConf, open_dict
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch
import omegaconf
import wandb

# def node_render(X):
#     """ Renders node features into CAD model. Assumes first 3 features are node types. """
#     num_node_types = 3  # [LINE, ARC, CIRCLE]
#     node_types = torch.argmax(X[..., :num_node_types], dim=-1)  # bs, n
    
#     # 根据 node_types， 绘制出相应的线段，然后从线段上均匀取100个点
#     bs, n = node_types.shape
#     X_rendered = torch.zeros((bs, n, 100, 3), device=X.device)  # 假设每个节点渲染成100个点
#     for b in range(bs):
#         for i in range(n):
#             t = node_types[b, i].item()
#             P0 = X[b, i, 3:6]
#             P1 = X[b, i, 6:9]
#             if t == 0:  # LINE
#                 points = torch.linspace(0, 1, steps=100, device=X.device).unsqueeze(-1)  # (100, 1)
#                 rendered_points = P0 + points * (P1 - P0)  # (100, 3)
#             elif t == 1:  # ARC
#                 C = X[b, i, 9:12]
#                 r = torch.norm(P0 - C)
#                 vec0 = P0 - C
#                 vec1 = P1 - C
#                 angle0 = torch.atan2(vec0[1], vec0[0])
#                 angle1 = torch.atan2(vec1[1], vec1[0])
#                 if angle1 < angle0:
#                     angle1 += 2 * torch.pi
#                 angles = torch.linspace(angle0, angle1, steps=100, device=X.device)
#                 rendered_points = C + r * torch.stack((torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), dim=-1)
#             else:  # CIRCLE
#                 C = X[b, i, 9:12]
#                 r = torch.norm(P0 - C)
#                 angles = torch.linspace(0, 2 * torch.pi, steps=100, device=X.device)
#                 rendered_points = C + r * torch.stack((torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), dim=-1)
#             X_rendered[b, i] = rendered_points
#     return X_rendered  # bs, n, 100, 3

def node_render(X):
    """
    X: (bs, n, d)
    first 3 dims = one-hot node type
    P0 = X[..., 3:6]
    P1 = X[..., 6:9]
    C  = X[..., 9:12]
    output: (bs, n, 100, 3)
    """
    device = X.device
    bs, n, _ = X.shape
    num_points = 100

    # node type = [LINE, ARC, CIRCLE]
    node_types = torch.argmax(X[..., :3], dim=-1)     # (bs, n)

    # geometry
    P0 = X[..., 3:6]          # (bs, n, 3)
    P1 = X[..., 6:9]          # (bs, n, 3)
    C  = X[..., 9:12]         # (bs, n, 3)

    # output
    X_render = torch.zeros((bs, n, num_points, 3), device=device)

    # t for line interpolation
    t = torch.linspace(0, 1, num_points, device=device)           # (100,)
    t = t.view(1, 1, num_points, 1)                               # (1,1,100,1)

    # =========================
    # LINE
    # =========================
    mask_line = (node_types == 0).view(bs, n, 1, 1)               # (bs,n,1,1)
    if mask_line.any():
        P0e = P0.unsqueeze(2)                                     # (bs,n,1,3)
        P1e = P1.unsqueeze(2)                                     # (bs,n,1,3)
        line_points = P0e + t * (P1e - P0e)                       # (bs,n,100,3)
        X_render = X_render.where(~mask_line, line_points)

    # =========================
    # ARC
    # =========================
    mask_arc = (node_types == 1).view(bs, n, 1, 1)
    if mask_arc.any():
        vec0 = P0 - C                                             # (bs,n,3)
        vec1 = P1 - C
        angle0 = torch.atan2(vec0[...,1], vec0[...,0])            # (bs,n)
        angle1 = torch.atan2(vec1[...,1], vec1[...,0])
        angle1 = angle1 + (angle1 < angle0) * (2 * torch.pi)

        # expand to (bs, n, 100)
        angles = angle0.unsqueeze(-1) + t.squeeze(-1) * (angle1 - angle0).unsqueeze(-1)

        r = torch.norm(vec0, dim=-1).unsqueeze(-1).unsqueeze(-1)  # (bs,n,1,1)
        Ce = C.unsqueeze(2)

        arc_points = torch.cat([
            (Ce[..., 0:1] + torch.cos(angles).unsqueeze(-1) * r),
            (Ce[..., 1:2] + torch.sin(angles).unsqueeze(-1) * r),
            Ce[..., 2:3].expand(bs, n, num_points, 1)
        ], dim=-1)

        X_render = X_render.where(~mask_arc, arc_points)

    # =========================
    # CIRCLE
    # =========================
    mask_circle = (node_types == 2).view(bs, n, 1, 1)
    if mask_circle.any():
        # angles must be (bs,n,100)
        base_angles = torch.linspace(0, 2 * torch.pi, num_points, device=device)   # (100,)
        base_angles = base_angles.view(1, 1, num_points).expand(bs, n, num_points)

        r = torch.norm(P0 - C, dim=-1).unsqueeze(-1)          # (bs,n,1)
        Ce = C.unsqueeze(2)                                    # (bs,n,1,3)

        circle_points = torch.cat([
            Ce[..., 0:1] + torch.cos(base_angles).unsqueeze(-1) * r.unsqueeze(-1),
            Ce[..., 1:2] + torch.sin(base_angles).unsqueeze(-1) * r.unsqueeze(-1),
            Ce[..., 2:3].expand(bs, n, num_points, 1)
        ], dim=-1)

        X_render = X_render.where(~mask_circle, circle_points)

    return X_render



def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs('graphs')
        os.makedirs('chains')
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs('graphs/' + args.general.name)
        os.makedirs('chains/' + args.general.name)
    except OSError:
        pass


def normalize(X, E, y, norm_values, norm_biases, node_mask, dataset_info=None):
    # zwh change
    X_type = X[..., :3]
    X_geom = X[..., 3:]
    geom_mean = dataset_info.geom_mean.to(X.device).view(1, 1, -1)
    geom_std  = dataset_info.geom_std.to(X.device).view(1, 1, -1)
    X_geom = (X_geom - geom_mean) / (geom_std + 1e-8)
    X = torch.cat([X_type, X_geom], dim=-1)
    
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0  # 消除自环

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False, dataset_info=None):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X_type = X[..., :3]
    X_geom = X[..., 3:]
    geom_mean = dataset_info.geom_mean.to(X.device).view(1, 1, -1)
    geom_std  = dataset_info.geom_std.to(X.device).view(1, 1, -1)
    X_geom = X_geom * (geom_std + 1e-8) + geom_mean
    X = torch.cat([X_type, X_geom], dim=-1)
    
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            if key not in cfg.model.keys():
                setattr(cfg.model, key, val)
    return cfg


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X[..., :3], dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': f'graph_ddm_{cfg.dataset.name}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')