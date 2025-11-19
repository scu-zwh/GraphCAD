import os
import torch
import pathlib
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

class CADGraphDataset(InMemoryDataset):
    """
    本地 CAD Graph 数据集
    每个 .pt 文件都是一个 torch_geometric.data.Data 对象
    不需要下载
    """
    def __init__(self, split, root, file_list_path, transform=None, pre_transform=None, pre_filter=None):
        self.split = split  # train / val / test
        self.file_list_path = file_list_path  # 例如 connected_graphs.txt
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # 不用实际下载，只是占位
        return [f"{self.split}.txt"]

    @property
    def processed_file_names(self):
        # 每个 split 会保存为单独的 .pt
        return [f"{self.split}.pt"]

    def download(self):
        # 本地数据集，不需要下载
        pass

    def process(self):
        print(f"Processing local CADGraphDataset split = {self.split}")

        # 读取包含 .pt 文件路径列表
        with open(self.file_list_path, "r") as f:
            graph_rel_paths = [line.strip() for line in f if line.strip()]

        data_list = []
        
        for i, rel_path in enumerate(tqdm(graph_rel_paths, desc=f"Loading {self.split} graphs")):
            full_path = os.path.join(self.root, f"{rel_path}.pt")
            if not os.path.exists(full_path):
                print(f"⚠️ 文件不存在: {full_path}")
                continue
            try:
                data = torch.load(full_path, weights_only=False)
                
                cad_geom = data.x[:, 1:].clone()  # 提取 CAD 几何信息
                edge_type_id = data.x[:, 0].long()
                data.x = F.one_hot(edge_type_id, num_classes=3).float()
                
                data.x = torch.cat([data.x, cad_geom], dim=-1)  # 拼接几何信息
                
                # 由于在创建 Graph 时缺少 edge_attr，自动补一个 dummy 特征
                if getattr(data, "edge_attr", None) is None:
                    num_edges = data.edge_index.shape[1]
                    data.edge_attr = torch.zeros((num_edges, 2), dtype=torch.float)
                    data.edge_attr[:, 1] = 1

                # ✅ 若缺少 y，则补一个空张量，shape = (1, 0)
                if getattr(data, "y", None) is None:
                    data.y = torch.zeros((1, 0), dtype=torch.float)

                # 常规预处理                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                    
                data.idx = i  # 添加索引属性
                    
                data_list.append(data)
            except Exception as e:
                print(f"⚠️ 加载失败 {full_path}: {e}")

        torch.save(self.collate(data_list), self.processed_paths[0])
        print(f"✅ 已处理 {len(data_list)} 个图, 保存至 {self.processed_paths[0]}")


class CADGraphDataModule(AbstractDataModule):
    """
    CAD Graph 数据模块
    封装 train/val/test 三个 Dataset
    """
    def __init__(self, cfg):
        self.cfg = cfg
        root_path = cfg.dataset.root_path  # 根路径，如 data/deepcad/cad_graph/
        
        # 三个 split 列表路径
        train_list = cfg.dataset.train_list
        val_list = cfg.dataset.val_list
        test_list = cfg.dataset.test_list

        datasets = {
            'train': CADGraphDataset(split='train', root=root_path, file_list_path=train_list),
            'val': CADGraphDataset(split='val', root=root_path, file_list_path=val_list),
            'test': CADGraphDataset(split='test', root=root_path, file_list_path=test_list)
        }
        
        # temp = datasets['train']
        # temp_data = temp[0]
        
        # print(temp_data)
        # for key in temp_data.keys:
        #     print(f"{key}: {temp_data[key]}")
        # exit(0)

        super().__init__(cfg, datasets)
        self.inner = datasets['train']
        self.datasets = datasets

    def __getitem__(self, item):
        return self.inner[item]
    
    def valency_count(self, max_n_nodes):
        """
        Compute degree (valency) distribution for CAD graphs.
        Return: tensor of length (3 * max_n_nodes - 2)
                following DiGress convention.
        """
        all_deg = []

        for data in tqdm(self.datasets['train'] + self.datasets['val'] + self.datasets['test'],
                        desc="Computing CAD valency distribution"):
            # PyG Graph data: edge_index shape (2, E)
            edge_index = data.edge_index
            deg = torch.bincount(edge_index[0], minlength=data.num_nodes)
            all_deg.append(deg)

        all_deg = torch.cat(all_deg, dim=0).float()

        # max allowed by DiGress (ensures shape compatibility)
        max_val = 3 * max_n_nodes - 2
        hist = torch.zeros(max_val)

        max_deg = int(all_deg.max().item())
        max_deg = min(max_deg, max_val - 1)

        for d in all_deg:
            idx = min(int(d.item()), max_val - 1)
            hist[idx] += 1

        hist = hist / hist.sum()  # normalize to probability

        return hist


class CADinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False, meta=None):
        self.name = 'cad'
        self.atom_decoder = ['LINE', 'ARC', 'SPLINE']
        self.atom_encoder = {t: i for i, t in enumerate(self.atom_decoder)}
        self.num_atom_types = 3
        
        self.geom_mean = torch.tensor([0.0514, -0.0298, 0.0908, 0.0611, -0.0821, 0.1319, 0.0033, -0.0050, 0.0153, 0.0338])
        self.geom_std = torch.tensor([0.4446, 0.3368, 0.3079, 0.4530, 0.3793, 0.3330, 0.1340, 0.1325, 0.1170, 0.1505])

        self.edge_density = torch.tensor(0.000459950853837654)

        self.n_nodes = torch.tensor([0.0000, 0.0000, 0.0000, 0.1163, 0.0000, 0.0000, 0.1085, 0.0000, 0.0000,
                                    0.0186, 0.0000, 0.0000, 0.1882, 0.0000, 0.0000, 0.0799, 0.0000, 0.0000,
                                    0.1399, 0.0000, 0.0000, 0.0426, 0.0000, 0.0000, 0.1653, 0.0000, 0.0000,
                                    0.0308, 0.0000, 0.0000, 0.0351, 0.0000, 0.0000, 0.0145, 0.0000, 0.0000,
                                    0.0416, 0.0000, 0.0000, 0.0070, 0.0000, 0.0000, 0.0095, 0.0000, 0.0000,
                                    0.0005, 0.0000, 0.0000, 0.0017])
        self.max_n_nodes = len(self.n_nodes) - 1

        self.node_types = torch.tensor([0.8556, 0.1444, 0.0000])

        # edge type 也通常只有 1 种（邻接）
        self.edge_types = torch.tensor([0.8263, 0.1737])

        # valency(节点度) 分布
        self.valency_distribution = torch.tensor([0.0000, 0.0861, 0.0430, 0.0020, 0.8688, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])

        # ---------------------------
        # Finalize initialization
        # ---------------------------
        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
        
        
def get_train_graph_signatures(cfg, train_dataloader, dataset_infos):
    """
    Returns a set of signatures of all training CAD graphs.
    用于 Novelty Metric：生成的 CAD 图是否不在训练集出现过。
    """

    from analysis.cad_utils import BasicCADMetrics  # 你需要根据文件路径调整导入
    cad_metrics = BasicCADMetrics(
        dataset_infos=dataset_infos  # ["SPLINE","ARC","LINE"]
    )

    signatures = set()

    for batch in train_dataloader:
        # batch 是 PyG Batch，可拆为 Data list
        data_list = batch.to_data_list()

        for g in data_list:
            sig = cad_metrics.make_signature(g.x)
            signatures.add(sig)

    return signatures
