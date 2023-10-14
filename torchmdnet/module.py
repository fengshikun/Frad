import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.nn.functional import mse_loss, l1_loss, smooth_l1_loss

from pytorch_lightning import LightningModule
from torchmdnet.models.model import create_model, load_model


class LNNP(LightningModule):
    def __init__(self, hparams, prior_model=None, mean=None, std=None):
        super(LNNP, self).__init__()
        self.save_hyperparameters(hparams)

        if self.hparams.load_model:
            self.model = load_model(self.hparams.load_model, args=self.hparams)
        elif self.hparams.pretrained_model:
            self.model = load_model(self.hparams.pretrained_model, args=self.hparams, mean=mean, std=std)
        else:
            self.model = create_model(self.hparams, prior_model, mean, std)

        # initialize exponential smoothing
        self.ema = None
        self._reset_ema_dict()

        # initialize loss collection
        self.losses = None
        self._reset_losses_dict()

        # seperate noisy node and finetune target
        self.sep_noisy_node = self.hparams.sep_noisy_node
        self.train_loss_type = self.hparams.train_loss_type

        if self.hparams.mask_atom:
            self.criterion = torch.nn.CrossEntropyLoss()
        
        self.bond_length_scale = self.hparams.bond_length_scale
        if self.bond_length_scale > 0:
            pass

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.lr_schedule == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, self.hparams.lr_cosine_length)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        elif self.hparams.lr_schedule == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                "min",
                factor=self.hparams.lr_factor,
                patience=self.hparams.lr_patience,
                min_lr=self.hparams.lr_min,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        else:
            raise ValueError(f"Unknown lr_schedule: {self.hparams.lr_schedule}")
        return [optimizer], [lr_scheduler]

    def forward(self, z, pos, batch=None, batch_org=None):
        return self.model(z, pos, batch=batch, batch_org=batch_org)

    def training_step(self, batch, batch_idx):
        if self.train_loss_type == 'smooth_l1_loss':
            return self.step(batch, smooth_l1_loss, 'train')
        return self.step(batch, mse_loss, "train")

    def validation_step(self, batch, batch_idx, *args):
        if len(args) == 0 or (len(args) > 0 and args[0] == 0):
            return self.step(batch, l1_loss, "val")
            # validation step
            return self.step(batch, mse_loss, "val")
        # test step
        return self.step(batch, l1_loss, "test")

    def test_step(self, batch, batch_idx):
        # res = self.step(batch, l1_loss, "test")
        # print(res)
        # return res
        return self.step(batch, l1_loss, "test")


    def process_batch_idx(self, batch):
        # process the idx of bond_target, angle_target and dihedral_target.
        batch_info = batch['batch']
        batch_num = batch._num_graphs

        slice_dict = batch._slice_dict
        bond_target_indx = slice_dict['bond_target']
        angle_target_indx = slice_dict['angle_target']
        dihedral_target_indx = slice_dict['dihedral_target']
        rotate_dihedral_target_indx = slice_dict['rotate_dihedral_target']
        for i in range(batch_num):
            cur_num = slice_dict['pos'][i] # add to bond idx
            
            batch.bond_target[bond_target_indx[i]:bond_target_indx[i+1]][:, :2] += cur_num
            batch.angle_target[angle_target_indx[i]:angle_target_indx[i+1]][:, :3] += cur_num
            batch.dihedral_target[dihedral_target_indx[i]:dihedral_target_indx[i+1]][:, :4] += cur_num
            batch.rotate_dihedral_target[rotate_dihedral_target_indx[i]:rotate_dihedral_target_indx[i+1]][:, :4] += cur_num
        

    def step(self, batch, loss_fn, stage):
        with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):
            # TODO: the model doesn't necessarily need to return a derivative once
            # Union typing works under TorchScript (https://github.com/pytorch/pytorch/pull/53180)
            if stage == 'test' and 'org_pos' in batch.keys:
                pred, noise_pred, deriv = self(batch.z, batch.org_pos, batch.batch) # use the org pos
            else:
                if self.sep_noisy_node:
                    pred, _, deriv = self(batch.z, batch.org_pos, batch.batch)
                    _, noise_pred, _ = self(batch.z, batch.pos, batch.batch)
                else:
                
                    if self.bond_length_scale > 0:
                        self.process_batch_idx(batch)
                        pred, noise_pred, deriv = self(batch.z, batch.pos, batch.batch, batch_org=batch)
                    else:
                        pred, noise_pred, deriv = self(batch.z, batch.pos, batch.batch)

        denoising_is_on = ("pos_target" in batch or "bond_target" in batch) and (self.hparams.denoising_weight > 0) and (noise_pred is not None)


        loss_y, loss_dy, loss_pos, mask_atom_loss = 0, 0, 0, 0


        # check whether mask, only in pretraining, deriv is logits
        if self.hparams.mask_atom:
            mask_indices = batch['masked_atom_indices']
            mask_logits = deriv[mask_indices]
            mask_atom_loss = self.criterion(mask_logits, batch.mask_node_label)
            self.losses[stage + "_mask_atom_loss"].append(mask_atom_loss.detach())

        if self.hparams.derivative:
            if "y" not in batch:
                # "use" both outputs of the model's forward function but discard the first
                # to only use the derivative and avoid 'Expected to have finished reduction
                # in the prior iteration before starting a new one.', which otherwise get's
                # thrown because of setting 'find_unused_parameters=False' in the DDPPlugin
                deriv = deriv + pred.sum() * 0

            # force/derivative loss
            loss_dy = loss_fn(deriv, batch.dy)

            if stage in ["train", "val"] and self.hparams.ema_alpha_dy < 1:
                if self.ema[stage + "_dy"] is None:
                    self.ema[stage + "_dy"] = loss_dy.detach()
                # apply exponential smoothing over batches to dy
                loss_dy = (
                    self.hparams.ema_alpha_dy * loss_dy
                    + (1 - self.hparams.ema_alpha_dy) * self.ema[stage + "_dy"]
                )
                self.ema[stage + "_dy"] = loss_dy.detach()

            if self.hparams.force_weight > 0:
                self.losses[stage + "_dy"].append(loss_dy.detach())

        if "y" in batch:
            if (noise_pred is not None) and not denoising_is_on:
                # "use" both outputs of the model's forward (see comment above).
                pred = pred + noise_pred.sum() * 0

            if batch.y.ndim == 1:
                batch.y = batch.y.unsqueeze(1)
            
            # if self.hparams["prior_model"] == "Atomref":
            #     batch.y = self.get_energy(batch)

            if torch.isnan(pred).sum():
                print('pred nan happends')
            # energy/prediction loss
            loss_y = loss_fn(pred, batch.y)
            if torch.isnan(loss_y).sum():
                print('loss nan happens')

            if stage in ["train", "val"] and self.hparams.ema_alpha_y < 1:
                if self.ema[stage + "_y"] is None:
                    self.ema[stage + "_y"] = loss_y.detach()
                # apply exponential smoothing over batches to y
                loss_y = (
                    self.hparams.ema_alpha_y * loss_y
                    + (1 - self.hparams.ema_alpha_y) * self.ema[stage + "_y"]
                )
                self.ema[stage + "_y"] = loss_y.detach()

            if self.hparams.energy_weight > 0:
                self.losses[stage + "_y"].append(loss_y.detach())

        if denoising_is_on:
            if "y" not in batch:
                if isinstance(noise_pred, list): # bond angle dihedral
                    noise_pred = [ele + pred.sum() * 0 for ele in noise_pred]
                else:
                # "use" both outputs of the model's forward (see comment above).
                    noise_pred = noise_pred + pred.sum() * 0
            
            def weighted_mse_loss(input, target, weight):
                return (weight.reshape(-1, 1).repeat((1, 3)) * (input - target) ** 2).mean()
            def mse_loss(input, target):
                return ((input - target) ** 2).mean()

            if 'wg' in batch.keys:
                loss_fn = weighted_mse_loss
                wt = batch['w1'].sum() / batch['idx'].shape[0]
                weights = batch['wg'] / wt
            else:
                loss_fn = mse_loss
            if self.model.pos_normalizer is not None:
                normalized_pos_target = self.model.pos_normalizer(batch.pos_target)
                if 'wg'in batch.keys:
                    loss_pos = loss_fn(noise_pred, normalized_pos_target, weights)
                else:
                    loss_pos = loss_fn(noise_pred, normalized_pos_target)
                self.losses[stage + "_pos"].append(loss_pos.detach())
            elif self.model.bond_pos_normalizer is not None:
                # bond, angle, dihedral
                normalized_bond_target = self.model.bond_pos_normalizer(batch.bond_target[:,-1])
                normalized_angle_target = self.model.angle_pos_normalizer(batch.angle_target[:,-1])
                normalized_dihedral_target = self.model.dihedral_pos_normalizer(batch.dihedral_target[:,-1])
                normalized_rotate_dihedral_target = self.model.rotate_dihedral_pos_normalizer(batch.rotate_dihedral_target[:,-1])
                loss_bond = loss_fn(noise_pred[0], normalized_bond_target)
                loss_angle = loss_fn(noise_pred[1], normalized_angle_target)
                # loss_angle = loss_fn(noise_pred[1], batch.angle_target[:,-1])
                
                loss_dihedral = loss_fn(noise_pred[2], normalized_dihedral_target)
                # loss_dihedral = loss_fn(noise_pred[2], batch.dihedral_target[:,-1])
                
                # loss_rotate_dihedral = loss_fn(noise_pred[3], batch.rotate_dihedral_target[:,-1])
                loss_rotate_dihedral = loss_fn(noise_pred[3], normalized_rotate_dihedral_target)
            

                self.losses[stage + "_bond"].append(loss_bond.detach())
                # self.losses[stage + "_angle"].append(loss_angle.detach() / (self.model.angle_pos_normalizer.std ** 2))
                self.losses[stage + "_angle"].append(loss_angle.detach())
                self.losses[stage + "_dihedral"].append(loss_dihedral.detach())
                # self.losses[stage + "_dihedral"].append(loss_dihedral.detach() / (self.model.dihedral_pos_normalizer.std ** 2))
                self.losses[stage + "_rotate_dihedral"].append(loss_rotate_dihedral.detach())
                # self.losses[stage + "_rotate_dihedral"].append(loss_rotate_dihedral.detach() / (self.model.rotate_dihedral_pos_normalizer.std ** 2))
                loss_pos = loss_bond + loss_angle + loss_dihedral + loss_rotate_dihedral
                if loss_pos.isnan().sum().item():
                    print('loss nan!!!')
            else:
                if 'wg'in batch.keys:
                    loss_pos = loss_fn(noise_pred, normalized_pos_target, weights)
                else:
                    loss_pos = loss_fn(noise_pred, normalized_pos_target)
            # loss_pos = loss_fn(noise_pred, normalized_pos_target)
                self.losses[stage + "_pos"].append(loss_pos.detach())

        # total loss
        loss = loss_y * self.hparams.energy_weight + loss_dy * self.hparams.force_weight + loss_pos * self.hparams.denoising_weight + mask_atom_loss

        self.losses[stage].append(loss.detach())

        # Frequent per-batch logging for training
        if stage == 'train':
            train_metrics = {k + "_per_step": v[-1] for k, v in self.losses.items() if (k.startswith("train") and len(v) > 0)}
            train_metrics['lr_per_step'] = self.trainer.optimizers[0].param_groups[0]["lr"]
            train_metrics['step'] = self.trainer.global_step   
            train_metrics['batch_pos_mean'] = batch.pos.mean().item()
            self.log_dict(train_metrics, sync_dist=True)

        return loss

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.hparams.lr_warmup_steps),
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def training_epoch_end(self, training_step_outputs):
        dm = self.trainer.datamodule
        if hasattr(dm, "test_dataset") and len(dm.test_dataset) > 0:
            should_reset = (
                self.current_epoch % self.hparams.test_interval == 0
                or (self.current_epoch - 1) % self.hparams.test_interval == 0
            )
            if should_reset:
                # reset validation dataloaders before and after testing epoch, which is faster
                # than skipping test validation steps by returning None
                self.trainer.reset_val_dataloader(self)

    def test_epoch_end(self, outputs):
        result_dict = {}
        if len(self.losses["test_y"]) > 0:
                result_dict["test_loss_y"] = torch.stack(
                    self.losses["test_y"]
                ).mean()
        return result_dict
    
    # TODO(shehzaidi): clean up this function, redundant logging if dy loss exists.
    def validation_epoch_end(self, validation_step_outputs):
        if not self.trainer.running_sanity_check:
            # construct dict of logged metrics
            result_dict = {
                "epoch": self.current_epoch,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "train_loss": torch.stack(self.losses["train"]).mean(),
                "val_loss": torch.stack(self.losses["val"]).mean(),
            }

            # add test loss if available
            if len(self.losses["test"]) > 0:
                result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()

            # if prediction and derivative are present, also log them separately
            if len(self.losses["train_y"]) > 0 and len(self.losses["train_dy"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
                result_dict["train_loss_dy"] = torch.stack(
                    self.losses["train_dy"]
                ).mean()
                result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
                result_dict["val_loss_dy"] = torch.stack(self.losses["val_dy"]).mean()

                if len(self.losses["test"]) > 0:
                    result_dict["test_loss_y"] = torch.stack(
                        self.losses["test_y"]
                    ).mean()
                    result_dict["test_loss_dy"] = torch.stack(
                        self.losses["test_dy"]
                    ).mean()

            if len(self.losses["train_y"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
            if len(self.losses['val_y']) > 0:
              result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
            if len(self.losses["test_y"]) > 0:
                result_dict["test_loss_y"] = torch.stack(
                    self.losses["test_y"]
                ).mean()

            # if denoising is present, also log it
            if len(self.losses["train_pos"]) > 0:
                result_dict["train_loss_pos"] = torch.stack(
                    self.losses["train_pos"]
                ).mean()

            if len(self.losses["val_pos"]) > 0:
                result_dict["val_loss_pos"] = torch.stack(
                    self.losses["val_pos"]
                ).mean()

            if len(self.losses["test_pos"]) > 0:
                result_dict["test_loss_pos"] = torch.stack(
                    self.losses["test_pos"]
                ).mean()

            self.log_dict(result_dict, sync_dist=True)
        self._reset_losses_dict()

    def _reset_losses_dict(self):
        self.losses = {
            "train": [],
            "val": [],
            "test": [],
            "train_y": [],
            "val_y": [],
            "test_y": [],
            "train_dy": [],
            "val_dy": [],
            "test_dy": [],
            "train_pos": [],
            "val_pos": [],
            "test_pos": [],
            "train_mask_atom_loss": [],
            "val_mask_atom_loss": [],
            "test_mask_atom_loss": [],
            
            "train_bond": [],
            "val_bond": [],
            "test_bond": [],

            "train_angle": [],
            "val_angle": [],
            "test_angle": [],

            "train_dihedral": [],
            "val_dihedral": [],
            "test_dihedral": [],

            "train_rotate_dihedral": [],
            "val_rotate_dihedral": [],
            "test_rotate_dihedral": [],

        }

    def _reset_ema_dict(self):
        self.ema = {"train_y": None, "val_y": None, "train_dy": None, "val_dy": None}
