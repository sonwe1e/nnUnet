import torch
from torch import autocast
import numpy as np
from torch.cuda.amp import autocast as dummy_context
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch.nn as nn
from nnunetv2.training.loss.compound_losses import (
    DC_and_Focal_loss,
    DC_and_CE_loss,
    DC_and_BCE_loss,
    DC_and_CE_and_self_loss,
)
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss

from typing import Union, Tuple, List
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.BresstCancerNet import (
    BresstCancerNet,
)
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.UXNet import UXNET
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.MedNeXt import (
    MedNeXt,
)
from monai.networks.nets import UNETR, SwinUNETR, AttentionUnet
from torch.optim.lr_scheduler import _LRScheduler
import math


class CustomLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        max_lr,
        epochs,
        pct_start,
        steps_per_epoch,
        anneal_strategy="linear",
        final_lr=1e-7,
        last_epoch=-1,
        stage_line=0.666,
    ):
        """
        自定义学习率调度器，实现两个阶段的warmup和decay。

        参数：
        - optimizer: 优化器
        - max_lr: 最大学习率（第一次warmup的目标学习率）
        - epochs: 总训练轮数
        - pct_start: warmup阶段占总阶段的比例
        - steps_per_epoch: 每个epoch的迭代步数
        - anneal_strategy: 衰减策略（"linear"或"cos"）
        - final_lr: 最终的学习率
        - last_epoch: 上一次更新的epoch
        """
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = epochs * steps_per_epoch
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.stage_line_step = self.total_steps * stage_line
        self.optimizer = optimizer

        # 计算每个阶段的总步数
        self.stage1_steps = self.stage_line_step
        self.stage2_steps = self.total_steps - self.stage_line_step

        # 计算每个阶段的warmup步数
        self.stage1_warmup_steps = int(self.stage1_steps * pct_start)
        self.stage2_warmup_steps = int(self.stage2_steps * pct_start)

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch

        if step_num <= self.stage_line_step:
            # 第一阶段
            if step_num < self.stage1_warmup_steps:
                # 第一阶段的warmup
                return [
                    self._compute_warmup_lr(
                        base_lr=0.0,
                        max_lr=self.max_lr,
                        step_num=step_num,
                        warmup_steps=self.stage1_warmup_steps,
                    )
                    for _ in self.optimizer.param_groups
                ]
            else:
                # 第一阶段的decay
                decay_step = step_num - self.stage1_warmup_steps
                decay_steps = self.stage1_steps - self.stage1_warmup_steps
                return [
                    self._compute_decay_lr(
                        max_lr=self.max_lr,
                        final_lr=self.final_lr,
                        step_num=decay_step,
                        decay_steps=decay_steps,
                    )
                    for _ in self.optimizer.param_groups
                ]
        else:
            # 第二阶段
            step_in_stage2 = step_num - self.stage_line_step
            if step_in_stage2 < self.stage2_warmup_steps:
                # 第二阶段的warmup，目标是max_lr的一半
                return [
                    self._compute_warmup_lr(
                        base_lr=self.final_lr,
                        max_lr=self.max_lr * 0.5,
                        step_num=step_in_stage2,
                        warmup_steps=self.stage2_warmup_steps,
                    )
                    for _ in self.optimizer.param_groups
                ]
            else:
                # 第二阶段的decay
                decay_step = step_in_stage2 - self.stage2_warmup_steps
                decay_steps = self.stage2_steps - self.stage2_warmup_steps
                return [
                    self._compute_decay_lr(
                        max_lr=self.max_lr * 0.5,
                        final_lr=self.final_lr,
                        step_num=decay_step,
                        decay_steps=decay_steps,
                    )
                    for _ in self.optimizer.param_groups
                ]

    def _compute_warmup_lr(self, base_lr, max_lr, step_num, warmup_steps):
        # 线性warmup
        return base_lr + (max_lr - base_lr) * (step_num / warmup_steps)

    def _compute_decay_lr(self, max_lr, final_lr, step_num, decay_steps):
        # 根据衰减策略计算学习率
        if self.anneal_strategy == "linear":
            return max_lr - (max_lr - final_lr) * (step_num / decay_steps)
        elif self.anneal_strategy == "cos":
            return final_lr + 0.5 * (max_lr - final_lr) * (
                1 + math.cos(math.pi * step_num / decay_steps)
            )
        else:
            raise ValueError("anneal_strategy must be 'linear' or 'cos'")


class nnUNetTrainerBreast(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
        exp_name: str = "",
    ):
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device, exp_name
        )
        self.num_epochs = 500
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        self.batch_size = 2
        self.initial_lr = 4e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False  # True

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.initial_lr,
            epochs=self.num_epochs,
            pct_start=0.06,
            steps_per_epoch=self.num_iterations_per_epoch,
            anneal_strategy="linear",
        )
        # lr_scheduler = CustomLRScheduler(
        #     optimizer,
        #     max_lr=self.initial_lr,
        #     epochs=self.num_epochs,
        #     pct_start=0.06,
        #     steps_per_epoch=self.num_iterations_per_epoch,
        #     anneal_strategy="cos",
        # )
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        self.network.train()
        # self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file("")
        self.print_to_log_file(f"Epoch {self.current_epoch}")
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=8)}"
        )
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log("lrs", self.optimizer.param_groups[0]["lr"], self.current_epoch)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            _, output = self.network(data)
            # del data
            l = self.loss(_, output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.lr_scheduler.step()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
            self.lr_scheduler.step()
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            _, output = self.network(data)
            del data
            l = self.loss(_, output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output.shape, device=output.device, dtype=torch.float32
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=mask
        )

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            "loss": l.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }

    def _build_loss(self):
        loss = DC_and_CE_and_self_loss(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            {},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array(
                [1 / (2**i) for i in range(len(deep_supervision_scales))]
            )
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        """
        This is where you build the architecture according to the plans. There is no obligation to use
        get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
        you want. Even ignore the plans and just return something static (as long as it can process the requested
        patch size)
        but don't bug us with your bugs arising from fiddling with this :-P
        This is the function that is called in inference as well! This is needed so that all network architecture
        variants can be loaded at inference time (inference will use the same nnUNetTrainer that was used for
        training, so if you change the network architecture during training by deriving a new trainer class then
        inference will know about it).

        If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
        > label_manager = plans_manager.get_label_manager(dataset_json)
        > label_manager.num_segmentation_heads
        (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
        the number of outputs is != the number of classes. Also there is the ignore label for which no output
        should be generated. label_manager takes care of all that for you.)

        """

        model = BresstCancerNet(
            num_input_channels,
            num_output_channels,
            deep_supervision=enable_deep_supervision,
        )
        # model = UNETR(num_input_channels, num_output_channels, img_size=(144, 144, 144))
        # model = SwinUNETR(
        #     img_size=(128, 128, 128),
        #     in_channels=num_input_channels,
        #     out_channels=num_output_channels,
        #     use_v2=True,
        # )
        # model = AttentionUnet(
        #     3,
        #     in_channels=num_input_channels,
        #     out_channels=num_output_channels,
        #     channels=[32, 64, 128, 256, 320],
        #     strides=[2, 2, 2, 2],
        # )
        # model = MedNeXt(
        #     in_channels=num_input_channels,
        #     n_channels=32,
        #     n_classes=num_output_channels,
        #     exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
        #     kernel_size=3,
        #     deep_supervision=False,
        #     do_res=True,
        #     do_res_up_down=True,
        #     block_counts=[3, 4, 4, 4, 4, 4, 4, 4, 3],
        #     checkpoint_style="outside_block",
        # )
        # model = UXNET(
        #     num_input_channels,
        #     num_output_channels,
        #     # deep_supervision=enable_deep_supervision,
        # )
        return model

    def set_deep_supervision_enabled(self, enabled: bool):
        pass
