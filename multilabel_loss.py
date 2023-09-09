from torch import nn
import numpy as np
import torch
class MultiLabelCircleLoss(nn.Module):
    def __init__(self, reduction="mean", inf=1e12):
        """CircleLoss of MultiLabel, 多个目标类的多标签分类场景，希望“每个目标类得分都不小于每个非目标类的得分”
        多标签分类的交叉熵(softmax+crossentropy推广, N选K问题), LSE函数的梯度恰好是softmax函数
        让同类相似度与非同类相似度之间拉开一定的margin。
          - 使同类相似度比最大的非同类相似度更大。
          - 使最小的同类相似度比最大的非同类相似度更大。
          - 所有同类相似度都比所有非同类相似度更大。
        urls: [将“softmax+交叉熵”推广到多标签分类问题](https://spaces.ac.cn/archives/7359)
        args:
            reduction: str, Specifies the reduction to apply to the output, 输出形式. 
                            eg.``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``
            inf: float, Minimum of maths, 无穷大.  eg. 1e12
        returns:
            Tensor of loss.
        examples:
            >>> label, logits = [[1, 1, 1, 1], [0, 0, 0, 1]], [[0, 1, 1, 0], [1, 0, 0, 1],]
            >>> label, logits = torch.tensor(label).float(), torch.tensor(logits).float()
            >>> loss = MultiLabelCircleLoss()(logits, label)
        """
        super(MultiLabelCircleLoss, self).__init__()
        self.reduction = reduction
        self.inf = inf  # 无穷大

    def forward(self, logits, labels):
        logits = (1 - 2 * labels) * logits              # <3, 4>
        logits_neg = logits - labels * self.inf         # <3, 4>
        logits_pos = logits - (1 - labels) * self.inf   # <3, 4>
        zeros = torch.zeros_like(logits[..., :1])       # <3, 1>
        logits_neg = torch.cat([logits_neg, zeros], dim=-1)  # <3, 5>
        logits_pos = torch.cat([logits_pos, zeros], dim=-1)  # <3, 5>
        neg_loss = torch.logsumexp(logits_neg, dim=-1)       # <3, >
        pos_loss = torch.logsumexp(logits_pos, dim=-1)       # <3, >
        loss = neg_loss + pos_loss
        if "mean" == self.reduction:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


