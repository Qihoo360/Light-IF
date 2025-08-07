import torch
import torch.nn.functional as F

from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer


class EntropyLossTrainer(CustomSeq2SeqTrainer):

    def _tokenwise_metrics(self, logits: torch.Tensor, labels: torch.Tensor):

        log_probs = F.log_softmax(logits, dim=-1)

        ce_tok = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        ent_tok = -(log_probs.exp() * log_probs).sum(-1)

        return ce_tok, ent_tok

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1. pop labels & forward
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        # 2. shift 对齐（B, L-1, V）
        logits = outputs.logits
        logits = logits[:, :-1].contiguous()
        labels = labels[:, 1:].contiguous()

        # 3. 有效 token mask
        loss_mask = labels != IGNORE_INDEX                    # [B, L-1]
        logits, labels = logits[loss_mask], labels[loss_mask]

        # 4. 逐 token 指标 & loss
        ce_tok, ent_tok = self._tokenwise_metrics(logits, labels)
        delta_tok = ce_tok - self.finetuning_args.entropy_coef * ent_tok

        # 5. 取前 token_ratio
        n_tot = delta_tok.numel()
        k = max(int(self.finetuning_args.token_ratio * n_tot), 1)

        assert k <= n_tot
        _, top_idx = torch.topk(delta_tok, k=k, largest=True, sorted=False)
        # loss = delta_tok[top_idx].mean()
        loss = ce_tok[top_idx].mean()

        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            with torch.no_grad():
                self.log({
                    "ce_all": ce_tok.mean().item(), "entropy_all": ent_tok.mean().item(),
                    "ce_sel": ce_tok[top_idx].mean().item(), "entropy_sel": ent_tok[top_idx].mean().item(),
                    "n_token_total": n_tot, "n_token_used": k
                    })

        return (loss, outputs) if return_outputs else loss


def build_entropy_loss_func(ft_args):
    """
    Returns a function that matches Trainer.compute_loss_func signature:
        (outputs, labels, num_items_in_batch=None) -> torch.Tensor
    """

    coef   = float(ft_args.entropy_coef)
    ratio  = float(ft_args.token_ratio)

    # ---------- 内部真正计算损失 ----------
    def entropy_loss(outputs, labels, num_items_in_batch=None):
        # outputs.logits: [B, L, V]
        logits = outputs.logits[:, :-1, :]
        labels = labels[:, 1:]

        # 有效 token mask
        mask = labels != IGNORE_INDEX
        logits = logits[mask]           # [N, V]
        labels = labels[mask]           # [N]

        # 交叉熵 & 熵
        log_probs = F.log_softmax(logits, dim=-1)        # [N, V]
        ce_tok    = -log_probs.gather(-1, labels[:, None]).squeeze(-1)  # [N]
        ent_tok   = -(log_probs.exp() * log_probs).sum(-1)                   # [N]

        delta_tok = ce_tok - coef * ent_tok                                   # [N]

        # Top-k 选择
        N = delta_tok.numel()
        k = max(int(ratio * N), 1)
        assert k <= N
        _, top_idx = torch.topk(delta_tok, k=k, largest=True, sorted=False)

        if num_items_in_batch is not None:
            loss = ce_tok[top_idx].sum() / int(num_items_in_batch * ratio)
        else:
            loss = ce_tok[top_idx].mean()

        return loss

    return entropy_loss
    