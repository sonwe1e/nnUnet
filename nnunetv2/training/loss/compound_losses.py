import torch
from nnunetv2.training.loss.dice import (
    SoftDiceLoss,
    MemoryEfficientSoftDiceLoss,
    SoftSkeletonRecallLoss,
)
from nnunetv2.training.loss.robust_ce_loss import (
    RobustCrossEntropyLoss,
    TopKLoss,
    RobustFocalLoss,
)
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
import torch.nn.functional as F


def custom_loss_function(prediction, label):
    """
    自定义损失函数。

    参数：
    - prediction: 张量，形状为 (B, C, D, H, W)
    - label: 张量，形状为 (B, 1, D, H, W)

    返回：
    - loss: 标量，损失值
    """
    # 获取预测结果的形状
    B, C, D, H, W = prediction.shape

    # 第一步：计算正样本标准特征
    # 获取正样本的掩码
    positive_mask = label.squeeze(1) == 1  # 形状：(B, D, H, W)

    # 提取正样本特征
    # 将预测结果重新排列以方便索引
    prediction_reshaped = prediction.permute(0, 2, 3, 4, 1)  # 形状：(B, D, H, W, C)
    positive_features = prediction_reshaped[positive_mask]  # 形状：(N_pos, C)

    # 计算正样本标准特征
    if positive_features.shape[0] > 0:
        standard_positive_feature = positive_features.mean(dim=0)  # 形状：(C,)
    else:
        # 如果没有正样本，避免出现错误，设置为零向量
        standard_positive_feature = torch.zeros(C, device=prediction.device)

    # 第二步：计算正样本特征与标准特征的相似度
    # 对特征进行归一化
    # positive_features_norm = F.normalize(positive_features, p=2, dim=1)
    # standard_positive_feature_norm = F.normalize(standard_positive_feature, p=2, dim=0)

    # 计算余弦相似度
    if positive_features.shape[0] > 0:
        positive_similarities = F.cosine_similarity(
            positive_features, standard_positive_feature.unsqueeze(0), dim=1
        )
        # 希望相似度尽可能大，因此损失为 (1 - 相似度)
        positive_loss = torch.mean(1 - positive_similarities)
    else:
        positive_loss = 0.0

    # 第三步：识别易错分区域并计算相似度
    # 创建卷积核用于膨胀操作
    kernel = torch.ones((1, 1, 3, 3, 3), device=prediction.device)
    padding = 1

    # 将正样本掩码转换为浮点型用于卷积
    positive_mask_float = positive_mask.float().unsqueeze(1)  # 形状：(B, 1, D, H, W)

    # 进行膨胀操作，迭代10次
    dilated_mask = positive_mask_float.clone()
    for _ in range(10):
        dilated_mask = F.conv3d(dilated_mask, kernel, padding=padding)
        dilated_mask = (dilated_mask > 0).float()

    # 易错分区域掩码：膨胀后的掩码减去原正样本掩码
    dilated_mask = dilated_mask.squeeze(1)  # 形状：(B, D, H, W)
    easily_misclassified_mask = (dilated_mask == 1) & (
        positive_mask == 0
    )  # 形状：(B, D, H, W)

    # 提取易错分区域的特征
    misclassified_features = prediction_reshaped[
        easily_misclassified_mask
    ]  # 形状：(N_mis, C)

    # 如果存在易错分特征，计算其与标准特征的相似度
    if misclassified_features.shape[0] > 0:
        # misclassified_features_norm = F.normalize(misclassified_features, p=2, dim=1)
        misclassified_similarities = F.cosine_similarity(
            misclassified_features,
            standard_positive_feature.unsqueeze(0),
            dim=1,
        )
        # 希望相似度尽可能小，因此损失为 ReLU(相似度)
        misclassified_loss = torch.mean(F.relu(misclassified_similarities))
    else:
        misclassified_loss = 0.0

    # 第四步：计算总损失
    total_loss = misclassified_loss + positive_loss

    return total_loss


def custom_loss_functionv2(prediction, label):
    """
    自定义损失函数。

    参数：
    - prediction: 张量，形状为 (B, C, D, H, W)
    - label: 张量，形状为 (B, 1, D, H, W)

    返回：
    - loss: 标量，损失值
    """
    # 获取预测结果的形状
    B, C, D, H, W = prediction.shape

    # 第一步：计算正样本标准特征
    # 获取正样本的掩码
    positive_mask = label.squeeze(1) == 1  # 形状：(B, D, H, W)

    # 将预测结果重新排列以方便索引
    prediction_reshaped = prediction.permute(0, 2, 3, 4, 1)  # 形状：(B, D, H, W, C)

    # 提取正样本特征
    positive_features = prediction_reshaped[positive_mask]  # 形状：(N_pos, C)

    # 计算正样本标准特征
    if positive_features.shape[0] > 0:
        standard_positive_feature = positive_features.mean(dim=0)  # 形状：(C,)
    else:
        # 如果没有正样本，避免出现错误，设置为零向量
        standard_positive_feature = torch.zeros(C, device=prediction.device)

    # 对标准正样本特征进行归一化
    standard_positive_feature_norm = F.normalize(standard_positive_feature, p=2, dim=0)

    # 第二步：计算正样本特征与标准特征的相似度
    if positive_features.shape[0] > 0:
        positive_features_norm = F.normalize(positive_features, p=2, dim=1)
        positive_similarities = F.cosine_similarity(
            positive_features_norm, standard_positive_feature_norm.unsqueeze(0), dim=1
        )
        # 希望相似度尽可能大，因此损失为 (1 - 相似度)
        positive_loss = torch.mean(1 - positive_similarities)
    else:
        positive_loss = 0.0

    # 第三步：识别易错分区域并计算相似度（与之前相同）
    # 创建卷积核用于膨胀操作
    kernel = torch.ones((1, 1, 3, 3, 3), device=prediction.device)
    padding = 1

    # 将正样本掩码转换为浮点型用于卷积
    positive_mask_float = positive_mask.float().unsqueeze(1)  # 形状：(B, 1, D, H, W)

    # 进行膨胀操作，迭代10次
    dilated_mask = positive_mask_float.clone()
    for _ in range(10):
        dilated_mask = F.conv3d(dilated_mask, kernel, padding=padding)
        dilated_mask = (dilated_mask > 0).float()

    # 易错分区域掩码：膨胀后的掩码减去原正样本掩码
    dilated_mask = dilated_mask.squeeze(1)  # 形状：(B, D, H, W)
    easily_misclassified_mask = (dilated_mask == 1) & (
        ~positive_mask
    )  # 形状：(B, D, H, W)

    # 提取易错分区域的特征
    misclassified_features = prediction_reshaped[
        easily_misclassified_mask
    ]  # 形状：(N_mis, C)

    # 如果存在易错分特征，计算其与标准特征的相似度
    if misclassified_features.shape[0] > 0:
        misclassified_features_norm = F.normalize(misclassified_features, p=2, dim=1)
        misclassified_similarities = F.cosine_similarity(
            misclassified_features_norm,
            standard_positive_feature_norm.unsqueeze(0),
            dim=1,
        )
        # 希望相似度尽可能小，因此损失为 ReLU(相似度)
        misclassified_loss = torch.mean(F.relu(misclassified_similarities))
    else:
        misclassified_loss = 0.0

    # 第四步：在负样本区域寻找与标准正样本特征最相似的像素位置

    # 获取负样本掩码
    negative_mask = label.squeeze(1) == 0  # 形状：(B, D, H, W)

    # 对预测特征进行归一化
    prediction_reshaped_norm = F.normalize(
        prediction_reshaped, p=2, dim=-1
    )  # 形状：(B, D, H, W, C)

    # 计算整个图像的余弦相似度映射
    cosine_similarity_map = torch.einsum(
        "bdhwc,c->bdhw", prediction_reshaped_norm, standard_positive_feature_norm
    )
    # 形状：(B, D, H, W)

    # 仅获取负样本区域的相似度
    negative_similarities = cosine_similarity_map[negative_mask]  # 形状：(N_neg,)

    # 检查是否存在负样本相似度
    if negative_similarities.numel() > 0:
        # 选择相似度最高的像素位置
        top_N = 250  # 可以根据需要调整
        if negative_similarities.numel() < top_N:
            top_N = negative_similarities.numel()

        sorted_similarities, indices = torch.topk(
            negative_similarities, top_N, largest=True
        )

        # 获取这些像素的位置
        negative_positions = negative_mask.nonzero(as_tuple=False)  # 形状：(N_neg, 4)
        top_positions = negative_positions[indices]  # 形状：(top_N, 4)

        # 创建掩码并在这些位置上标记为1
        high_similarity_mask = torch.zeros_like(negative_mask, dtype=torch.float)
        high_similarity_mask[
            top_positions[:, 0],
            top_positions[:, 1],
            top_positions[:, 2],
            top_positions[:, 3],
        ] = 1.0

        # 对这些位置进行膨胀操作，迭代25次（每次膨胀增加2个像素，总共约50个像素）
        dilated_high_similarity_mask = high_similarity_mask.unsqueeze(
            1
        )  # 形状：(B, 1, D, H, W)
        for _ in range(10):
            dilated_high_similarity_mask = F.conv3d(
                dilated_high_similarity_mask, kernel, padding=padding
            )
            dilated_high_similarity_mask = (dilated_high_similarity_mask > 0).float()
        dilated_high_similarity_mask = dilated_high_similarity_mask.squeeze(
            1
        )  # 形状：(B, D, H, W)

        # 确保膨胀后的区域不包含正样本区域
        final_negative_regions = (dilated_high_similarity_mask == 1) & (~positive_mask)

        # 提取这些区域的特征
        negative_dilated_features = prediction_reshaped_norm[final_negative_regions]

        # 计算与标准正样本特征的相似度
        if negative_dilated_features.shape[0] > 0:
            negative_dilated_similarities = F.cosine_similarity(
                negative_dilated_features,
                standard_positive_feature_norm.unsqueeze(0),
                dim=1,
            )
            # 希望相似度尽可能小，因此损失为 ReLU(相似度)
            negative_dilated_loss = torch.mean(F.relu(negative_dilated_similarities))
        else:
            negative_dilated_loss = 0.0
    else:
        negative_dilated_loss = 0.0

    # 第五步：计算总损失
    total_loss = positive_loss + misclassified_loss + negative_dilated_loss

    return total_loss


class DC_and_CE_and_self_loss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        weight_ce=1,
        weight_dice=1,
        ignore_label=None,
        dice_class=SoftDiceLoss,
    ):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_and_self_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(
        self, feature: torch.Tensor, net_output: torch.Tensor, target: torch.Tensor
    ):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one hot encoded target variables "
                "(DC_and_CE_loss)"
            )
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = (
            self.dc(net_output, target_dice, loss_mask=mask)
            if self.weight_dice != 0
            else 0
        )
        ce_loss = (
            self.ce(net_output, target[:, 0])
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0)
            else 0
        )
        self_loss = custom_loss_functionv2(feature, target)

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + 5 * self_loss
        return result


class DC_and_CE_loss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        weight_ce=1,
        weight_dice=1,
        ignore_label=None,
        dice_class=SoftDiceLoss,
    ):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one hot encoded target variables "
                "(DC_and_CE_loss)"
            )
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = (
            self.dc(net_output, target_dice, loss_mask=mask)
            if self.weight_dice != 0
            else 0
        )
        ce_loss = (
            self.ce(net_output, target[:, 0])
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0)
            else 0
        )

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_Focal_loss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        weight_ce=1,
        weight_dice=1,
        ignore_label=None,
        dice_class=SoftDiceLoss,
    ):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_Focal_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustFocalLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one hot encoded target variables "
                "(DC_and_CE_loss)"
            )
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = (
            self.dc(net_output, target_dice, loss_mask=mask)
            if self.weight_dice != 0
            else 0
        )
        ce_loss = (
            self.ce(net_output, target[:, 0])
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0)
            else 0
        )

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_SkelREC_and_CE_loss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs,
        soft_skelrec_kwargs,
        ce_kwargs,
        weight_ce=1,
        weight_dice=1,
        weight_srec=1,
        ignore_label=None,
        dice_class=MemoryEfficientSoftDiceLoss,
    ):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_SkelREC_and_CE_loss, self).__init__()

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_srec = weight_srec
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.srec = SoftSkeletonRecallLoss(
            apply_nonlin=softmax_helper_dim1, **soft_skelrec_kwargs
        )

    def forward(
        self, net_output: torch.Tensor, target: torch.Tensor, skel: torch.Tensor
    ):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        srec_loss = self.srec(net_output, skel) if self.weight_srec != 0 else 0
        ce_loss = (
            (self.ce(net_output, target[:, 0].long())).mean()
            if self.weight_ce != 0
            else 0
        )

        result = (
            self.weight_ce * ce_loss
            + self.weight_dice * dc_loss
            + self.weight_srec * srec_loss
        )
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(
        self,
        bce_kwargs,
        soft_dice_kwargs,
        weight_ce=1,
        weight_dice=1,
        use_ignore_label: bool = False,
        dice_class=MemoryEfficientSoftDiceLoss,
    ):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs["reduction"] = "none"

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(
                mask.sum(), min=1e-8
            )
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(
        self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None
    ):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one hot encoded target variables "
                "(DC_and_CE_loss)"
            )
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = (
            self.dc(net_output, target_dice, loss_mask=mask)
            if self.weight_dice != 0
            else 0
        )
        ce_loss = (
            self.ce(net_output, target)
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0)
            else 0
        )

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_and_self_loss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        weight_ce=1,
        weight_dice=1,
        ignore_label=None,
        dice_class=SoftDiceLoss,
    ):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_topk_and_self_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(
        self, feature: torch.Tensor, net_output: torch.Tensor, target: torch.Tensor
    ):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one hot encoded target variables "
                "(DC_and_CE_loss)"
            )
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = (
            self.dc(net_output, target_dice, loss_mask=mask)
            if self.weight_dice != 0
            else 0
        )
        ce_loss = (
            self.ce(net_output, target)
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0)
            else 0
        )
        self_loss = custom_loss_functionv2(feature, target)

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + 5 * self_loss
        return result
