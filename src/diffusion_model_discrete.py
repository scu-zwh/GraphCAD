import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os

from models.transformer_model import GraphTransformer
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition
from src.diffusion import diffusion_utils
from metrics.train_metrics import TrainLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from src import utils

class CNNEncoder(nn.Module):
    def __init__(self, output_dim):
        super(CNNEncoder, self).__init__()

        # 输入: (bs*n, 100, 3)  → reshape → (bs*n, 3, 100)
        self.conv1 = nn.Conv1d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)

        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        """
        x: [bs, n, 100, 3]
        return: [bs, n, output_dim]
        """
        bs, n, point_num, _ = x.shape

        # Reshape for CNN
        x = x.view(bs * n, point_num, 3)      # (bs*n, 100, 3)
        x = x.permute(0, 2, 1)                # (bs*n, 3, 100)

        # Conv1D blocks
        x = F.relu(self.conv1(x))             # (bs*n, 32, 100)
        x = F.relu(self.conv2(x))             # (bs*n, 64, 100)
        x = F.relu(self.conv3(x))             # (bs*n, 128, 100)

        # Global max pooling over points
        x = torch.max(x, dim=-1)[0]           # (bs*n, 128)

        # FC to output_dim
        x = self.fc(x)                        # (bs*n, output_dim)

        # reshape back to (bs, n, output_dim)
        x = x.view(bs, n, -1)
        return x


class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train, dataset_infos)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.point_encoder = CNNEncoder(output_dim=64)  # 11.27 zwh change
        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        elif cfg.model.transition == 'marginal':

            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        # 归一化
        normalized_data = utils.normalize(X, E, data.y, [1,1,1], [0,0,0], node_mask, self.dataset_info)
        noisy_data = self.apply_noise(normalized_data.X, normalized_data.E, normalized_data.y, node_mask)
        
        # noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=X, true_E=E, true_y=data.y,
                               log=i % self.log_every_steps == 0)

        self.train_metrics(pred_X=pred.X, pred_E=pred.E, true_X=X, true_E=E, node_mask=node_mask,
                           log=i % self.log_every_steps == 0)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Size of the input features", self.Xdim, self.Edim, self.ydim)
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE'] :.3f}"
                      f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
                      f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
                      f" -- {time.time() - self.start_epoch_time:.1f}s ")
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}")
        if torch.cuda.is_available():
            print(torch.cuda.memory_summary())
        else:
            print("CUDA is not available. Skipping memory summary.")

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.sampling_metrics.reset()

    def validation_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        
        X, E = dense_data.X, dense_data.E
        # 归一化
        normalized_data = utils.normalize(X, E, data.y, [1,1,1], [0,0,0], node_mask, self.dataset_info)
        noisy_data = self.apply_noise(normalized_data.X, normalized_data.E, normalized_data.y, node_mask)
        
        # noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y,  node_mask, test=False)
        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(), self.val_X_kl.compute() * self.T, self.val_E_kl.compute() * self.T,
                   self.val_X_logp.compute(), self.val_E_logp.compute()]
        if wandb.run:
            wandb.log({"val/epoch_NLL": metrics[0],
                       "val/X_kl": metrics[1],
                       "val/E_kl": metrics[2],
                       "val/X_logp": metrics[3],
                       "val/E_logp": metrics[4]}, commit=False)

        self.print(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Atom type KL {metrics[1] :.2f} -- ",
                   f"Val Edge type KL: {metrics[2] :.2f}")

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))

        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save
            
            debug_path = "/mnt/data/zhengwenhao/workspace/DiGress/outputs/debug.txt"

            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(f"samples_left_to_generate: {samples_left_to_generate}\n")
                f.write(f"samples_left_to_save: {samples_left_to_save}\n")
                f.write(f"chains_left_to_save: {chains_left_to_save}\n")


            samples = []

            ident = 0
            while samples_left_to_generate > 0:
                bs = 2 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                samples.extend(self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                 save_final=to_save,
                                                 keep_chain=chains_save,
                                                 number_chain_steps=self.number_chain_steps))
                ident += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save
            self.print("Computing sampling metrics...")
            self.sampling_metrics(samples, self.name, self.current_epoch, val_counter=-1, test=False)
            self.print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
            print("Validation epoch end ends...")

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def test_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        # 归一化
        normalized_data = utils.normalize(X, E, data.y, [1,1,1], [0,0,0], node_mask, self.dataset_info)
        noisy_data = self.apply_noise(normalized_data.X, normalized_data.E, normalized_data.y, node_mask)
        # noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=True)
        return {'loss': nll}

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(), self.test_X_kl.compute(), self.test_E_kl.compute(),
                   self.test_X_logp.compute(), self.test_E_logp.compute()]
        if wandb.run:
            wandb.log({"test/epoch_NLL": metrics[0],
                       "test/X_kl": metrics[1],
                       "test/E_kl": metrics[2],
                       "test/X_logp": metrics[3],
                       "test/E_logp": metrics[4]}, commit=False)

        self.print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- ",
                   f"Test Edge type KL: {metrics[2] :.2f}")

        test_nll = metrics[0]
        if wandb.run:
            wandb.log({"test/epoch_NLL": test_nll}, commit=False)

        self.print(f'Test loss: {test_nll :.4f}')

        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        samples = []
        id = 0
        while samples_left_to_generate > 0:
            self.print(f'Samples left to generate: {samples_left_to_generate}/'
                       f'{self.cfg.general.final_model_samples_to_generate}', end='')
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            samples.extend(self.sample_batch(id, to_generate, num_nodes=None, save_final=to_save,
                                             keep_chain=chains_save, number_chain_steps=self.number_chain_steps))
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save
        self.print("Saving the generated graphs")
        filename = f'generated_samples1.txt'
        for i in range(2, 10):
            if os.path.exists(filename):
                filename = f'generated_samples{i}.txt'
            else:
                break
        with open(filename, 'w') as f:
            for item in samples:
                f.write(f"N={item[0].shape[0]}\n")
                atoms = item[0].tolist()
                f.write("X: \n")
                for at in atoms:
                    f.write(f"{at} ")
                f.write("\n")
                f.write("E: \n")
                for bond_list in item[1]:
                    for bond in bond_list:
                        f.write(f"{bond} ")
                    f.write("\n")
                f.write("\n")
        self.print("Generated graphs Saved. Computing sampling metrics...")
        self.sampling_metrics(samples, self.name, self.current_epoch, self.val_counter, test=True)
        self.print("Done testing.")


    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        bs, n, d = X.shape
        device = X.device

        # -------------------------------------------------------------
        # 1. Split X into (type, geom)
        # -------------------------------------------------------------
        X_type = X[..., :3]          # one-hot
        X_geom = X[..., 3:]          # continuous (bs,n,10)

        # -------------------------------------------------------------
        # 2. Get alpha_bar_T from discrete noise schedule
        # -------------------------------------------------------------
        ones = torch.ones((X.size(0), 1), device=device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        # reshape for broadcast
        alpha_T_bar = alpha_t_bar.view(bs, 1, 1)                    # (bs,1,1)
        sqrt_ab = torch.sqrt(alpha_T_bar)                           # √ᾱ_T
        var_T = 1.0 - alpha_T_bar                                   # variance
        var_T = torch.clamp(var_T, min=1e-8)
        log_var_T = torch.log(var_T)

        # -------------------------------------------------------------
        # 3. KL for continuous geometry dims  (Gaussian -> Gaussian)
        # -------------------------------------------------------------
        # μ = sqrt(alpha_bar) * x0
        mu_T = sqrt_ab * X_geom                       # (bs,n,10)

        # σ^2 = 1 - alpha_bar
        sigma2_T = var_T                              # (bs,1,1)

        # Gaussian KL:
        # 0.5 * [ μ^2 + σ^2 - 1 - log σ^2 ]
        kl_geom = 0.5 * (
            (mu_T ** 2) +
            sigma2_T -
            1.0 -
            log_var_T
        )

        # mask padded nodes
        kl_geom = kl_geom * node_mask.unsqueeze(-1)

        # sum over geom dims and nodes
        kl_geom = kl_geom.sum(dim=[1, 2])             # (bs,)

        # -------------------------------------------------------------
        # 4. KL for discrete dims (type + edges)  (original DiGress)
        # -------------------------------------------------------------
        # transition matrix at T
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device)

        # Compute transition probabilities
        probX_type = X_type @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        # assert probX.shape == X.shape

        limit_X = self.limit_dist.X[None, None, :3].expand(bs, n, -1).type_as(probX_type)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX_type,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)

        kl_type = F.kl_div(probX.log(), limit_dist_X, reduction='none')
        
        # kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')

        return diffusion_utils.sum_except_batch(kl_type) + \
               diffusion_utils.sum_except_batch(kl_distance_E) + kl_geom

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        # pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_X_type = F.softmax(pred.X[..., :3], dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        X_type = X[..., :3]
        prob_true = diffusion_utils.posterior_distributions(X=X_type, E=E, y=y, X_t=noisy_data['X_t'][..., :3], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb,
                                                            noisy_data=noisy_data)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X_type, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'][..., :3], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb,
                                                            noisy_data=noisy_data)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, node_mask):
        bs, n, dx = X.shape
        device = X.device

        # ------------------------------
        # 1. Split X into type + geom
        # ------------------------------
        X_type = X[..., :3]          # one-hot
        X_geom = X[..., 3:]          # continuous (10 dims)        
        
        # ------------------------------
        # 2. Compute noise values for t = 0.  加噪，同apply_noise
        # ------------------------------
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)
        
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_zeros)   # (bs, 1)

        # DDPM formulas:
        # alpha = sqrt(alpha_bar)
        # sigma = sqrt(1 - alpha_bar)
        alpha_t = torch.sqrt(alpha_t_bar).view(-1, 1, 1)                # (bs,1,1)
        sigma_t = torch.sqrt(1.0 - alpha_t_bar).view(-1, 1, 1)          # (bs,1,1)

        # Gaussian noise for geom dims
        eps_geom = torch.randn_like(X_geom) * node_mask.unsqueeze(-1)   # (bs,n,10)
        X0_geom = alpha_t * X_geom + sigma_t * eps_geom                # (bs,n,10)

        probX0_type = X_type @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0_type, probE=probE0, node_mask=node_mask)

        X0_type = F.one_hot(sampled0.X, num_classes=X_type.shape[-1]).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y
        # assert (X.shape == X0.shape) and (E.shape == E0.shape)
        
        X0 = torch.cat([X0_type, X0_geom], dim=-1)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # ------------------------------
        # 3. Predictions at t = 0
        # ------------------------------ 
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)       
        pred0 = self.forward(noisy_data, extra_data, node_mask)

        # ------------------------------
        # 4. Normalize predictions
        #    - X: 只对前 3 维做 softmax，其余几何维度填 1（不参与 logp）
        #    - E, y: 和原版一样
        # ------------------------------
        # 类型部分 softmax
        probX0_type = F.softmax(pred0.X[..., :3], dim=-1)
        # 构造一个和 pred0.X 同 shape 的张量
        probX0 = torch.ones_like(pred0.X)
        probX0[..., :3] = probX0_type  # 只前3维是真正的概率，后10维为1
        
        # probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # ------------------------------
        # 5. Mask 掉 padding 部分
        # ------------------------------
        # 节点 mask
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, node_mask):
        """
        Mixed diffusion for CAD:
        - X_type (first 3 dims): discrete diffusion
        - X_geom (last 10 dims): continuous Gaussian diffusion
        - E (2 dims): discrete diffusion
        """
        
        bs, n, dx = X.shape
        device = X.device

        # --------------------------------------------------------
        # 1. Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        # --------------------------------------------------------
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        # --------------------------------------------------------
        # 2. Split X into (type, geom)
        # --------------------------------------------------------
        X_type = X[..., :3]          # (bs,n,3)
        X_geom = X[..., 3:]          # (bs,n,10)

        # --------------------------------------------------------
        # 3. Discrete diffusion for node type & edges
        # --------------------------------------------------------
        # Q_t for categorical diffusion
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X_type @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t_type = F.one_hot(sampled_t.X, num_classes=Qtb.X.size(-1))
        # X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        # E_t = E_t.type_as(X)  # keep dtype consistent
        # assert (X.shape == X_t.shape) and (E.shape == E_t.shape)
        
        # --------------------------------------------------------
        # 4. Continuous diffusion for geom dims
        # --------------------------------------------------------
        # Use predefined discrete schedule for continuous Gaussian diffusion
        # beta_t = self.noise_schedule(t_normalized=t_float)              # (bs, 1)
        # alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)   # (bs, 1)

        # DDPM formulas:
        # alpha = sqrt(alpha_bar)
        # sigma = sqrt(1 - alpha_bar)
        alpha_t = torch.sqrt(alpha_t_bar).view(-1, 1, 1)                # (bs,1,1)
        sigma_t = torch.sqrt(1.0 - alpha_t_bar).view(-1, 1, 1)          # (bs,1,1)

        # Gaussian noise for geom dims
        eps_geom = torch.randn_like(X_geom) * node_mask.unsqueeze(-1)   # (bs,n,10)

        # forward diffusion for continuous part
        X_t_geom = alpha_t * X_geom + sigma_t * eps_geom                 # (bs,n,10)

        # --------------------------------------------------------
        # 5. Combine back
        # --------------------------------------------------------
        X_t = torch.cat([X_t_type, X_t_geom], dim=-1)

        # --------------------------------------------------------
        # 6. Mask + return
        # --------------------------------------------------------
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, node_mask)

        loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log())

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch

        if wandb.run:
            wandb.log({"kl prior": kl_prior.mean(),
                       "Estimator loss terms": loss_all_t.mean(),
                       "log_pn": log_pN.mean(),
                       "loss_term_0": loss_term_0,
                       'batch_test_nll' if test else 'val_nll': nll}, commit=False)
        return nll

    def forward(self, noisy_data, extra_data, node_mask):
        # 11.27 zwh change
        # 先将noisy_data render，然后随机取100个点
        from src.utils import node_render
        
        X_render = node_render(noisy_data['X_t'])   # bs, n, 100, 3
        X_encoding = self.point_encoder(X_render)  # bs, n, 64
        # print(X_encoding.shape)
        
        X = torch.cat((noisy_data['X_t'], X_encoding, extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        
        # Build the masks 哪些位置是真节点，哪些是 padding
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        node_mask_float = node_mask.float()
        
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X_type, E, y = z_T.X, z_T.E, z_T.y  # X_type: [bs, n_max, 3]
        
        X_geom = torch.randn((batch_size, n_max, self.Xdim_output - X_type.shape[-1]), device=self.device) * node_mask_float.unsqueeze(-1)

        X = torch.cat([X_type.float(), X_geom], dim=-1)

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)
        
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.  反向扩散主循环 从 T 到 0
        # 构建 tqdm 迭代器
        show_pbar = True
        iter_range = range(0, self.T)
        if show_pbar:
            from tqdm import tqdm
            iter_range = tqdm(iter_range, desc="Sampling reverse diffusion", leave=False)
            
            
        for s_int in reversed(iter_range):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        # sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
        
        # x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        # e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        # e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1
        
        # X_type = torch.argmax(X[..., :3], dim=-1)
        # X_geom = X[..., 3:]
        # X = torch.cat([F.one_hot(X_type, num_classes=3).float(), X_geom], dim=-1)
        # E = torch.argmax(E, dim=-1)
    
        # X[node_mask == 0] = - 1
        # E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1    

        # Prepare the chain for saving
        if keep_chain > 0:
            x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
            e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
            e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1
            final_X_chain = torch.argmax(X[..., :3], dim=-1)
            final_E_chain = torch.argmax(E, dim=-1)

            final_X_chain[node_mask == 0] = - 1
            final_E_chain[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1                       
            
            final_X_chain = final_X_chain[:keep_chain]
            final_E_chain = final_E_chain[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        # Visualize chains
        # if self.visualization_tools is not None:
        if False:
            self.print('Visualizing chains...')
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)       # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                         f'epoch{self.current_epoch}/'
                                                         f'chains/molecule_{batch_id + i}')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(result_path,
                                                                 chain_X[:, i, :].numpy(),
                                                                 chain_E[:, i, :].numpy())
                self.print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)
            self.print('\nVisualizing molecules...')

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dx = X_t.shape
        device = X_t.device

        # ------------------------------------------------
        # 0. 拆分 X_t = (类型, 几何)
        # ------------------------------------------------
        type_dim = 3
        geom_dim = dx - type_dim

        X_t_type = X_t[..., :type_dim]      # (bs, n, 3), one-hot
        X_t_geom = X_t[..., type_dim:]      # (bs, n, 10)

        # ------------------------------------------------
        # 1. 取噪声调度参数：beta_t, alphā_s, alphā_t
        # ------------------------------------------------        
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # 对应一步的 α_t = 1 - β_t
        alpha_t_step = 1.0 - beta_t                                 # (bs, 1)

        # ------------------------------------------------
        # 2. 计算转移矩阵（离散部分用）
        # ------------------------------------------------        
        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # ------------------------------------------------
        # 3. 网络预测（基于当前 z_t = (X_t, E_t, y_t)）
        # ------------------------------------------------
        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        # pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        # 类型 logits -> 概率
        pred_type_logits = pred.X[..., :type_dim]       # (bs, n, 3)
        pred_type_probs = F.softmax(pred_type_logits, dim=-1)

        # 几何部分预测（我们把它当作 x0 的预测）
        pred_geom_x0 = pred.X[..., type_dim:]          # (bs, n, 10)        
        
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        # ------------------------------------------------
        # 4. 离散部分 posterior：类型 + 边（完全沿用 DiGress）
        # ------------------------------------------------
        # 对类型：用 X_t_type 而不是整个 X_t
        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t_type,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        # 对边：完全照旧
        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        
        # ------------ 节点类型 posterior ------------
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_type_probs.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        # ------------ 边 posterior ------------
        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        # 从 posterior 采样离散类型 & 边
        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)      
        # 类型 index：(bs, n)，范围 [0, x_classes-1]  one-hot 类型
        X_s_type = F.one_hot(sampled_s.X, num_classes=type_dim).float()   # (bs, n, 3)
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        # ------------------------------------------------
        # 5. 连续几何部分：DDPM posterior（Gaussian）
        # ------------------------------------------------
        # 按 x0-parameterization 公式：
        # μ = c1 * x0_pred + c2 * x_t
        # σ^2 = (1 - ᾱ_s)/(1 - ᾱ_t) * β_t

        # reshape 方便广播
        alpha_s_bar = alpha_s_bar.view(bs, 1, 1)     # (bs,1,1)
        alpha_t_bar = alpha_t_bar.view(bs, 1, 1)     # (bs,1,1)
        beta_t = beta_t.view(bs, 1, 1)               # (bs,1,1)
        alpha_t_step = alpha_t_step.view(bs, 1, 1)   # (bs,1,1)

        # 分母 1 - ᾱ_t
        one_minus_alpha_t_bar = 1.0 - alpha_t_bar
        one_minus_alpha_t_bar = torch.clamp(one_minus_alpha_t_bar, min=1e-8)

        # 系数
        # 参考 DDPM：c1 = sqrt(ᾱ_{t-1}) * β_t / (1 - ᾱ_t)
        #           c2 = sqrt(α_t) * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        c1 = torch.sqrt(alpha_s_bar) * beta_t / one_minus_alpha_t_bar
        c2 = torch.sqrt(alpha_t_step) * (1.0 - alpha_s_bar) / one_minus_alpha_t_bar

        mu_geom = c1 * pred_geom_x0 + c2 * X_t_geom   # (bs, n, 10)

        var_geom = (1.0 - alpha_s_bar) / one_minus_alpha_t_bar * beta_t
        var_geom = torch.clamp(var_geom, min=1e-8)
        sigma_geom = torch.sqrt(var_geom)             # (bs,1,1)

        # 采样高斯噪声（只对真实节点，加 mask）
        eps = torch.randn_like(X_t_geom) * node_mask.unsqueeze(-1).float()
        X_s_geom = mu_geom + sigma_geom * eps         # (bs, n, 10)

        # ------------------------------------------------
        # 6. 拼回来：完整 X_s = [类型 one-hot, 几何连续]
        # ------------------------------------------------
        X_s = torch.cat([X_s_type, X_s_geom], dim=-1)   # (bs, n, 3+10)

        # 保证对称
        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        # assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0).to(device))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0).to(device))

        return out_one_hot.mask(node_mask), out_discrete.mask(node_mask, collapse=True)

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
