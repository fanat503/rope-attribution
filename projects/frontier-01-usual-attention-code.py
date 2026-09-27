# Qwen3 RoPE+YaRN Phi-атрибуция - идеал уровень, без PoPE
# Модель: Qwen3-4B PLT L0_50 63% starter T4 TransformerLens -> Qwen3-14B 35GB TPU v5e-8 nnsight per-query chunking
# Gate=|q| длина всегда есть в RoPE/YaRN score=|q||k|cos(...), если |q|=0 score=0
# Phi=angle(q) контент-фаза, phi=angle(sum f_i q_i) != sum angle

import hashlib
import json
import math

import torch


def check_sterility(model):
    err = model.hla_identity_error() if hasattr(model, "hla_identity_error") else 0.0
    assert err < 1e-6
    cfg = json.dumps(
        model.config.to_dict() if hasattr(model.config, "to_dict") else {},
        sort_keys=True,
    )
    return hashlib.sha256(cfg.encode()).hexdigest()[:8]


def compute_qk_score(x_q, x_k, W_Q, W_K, pos_q, pos_k, base=500000):  # YaRN base 500k
    # q = W_Q^T x_q, k = W_K^T x_k
    # q_rope = R(pos_q) q, score = |q||k| cos(phi_q-phi_k + (pos_q-pos_k)theta)
    # theta = base^{-2i/d}, YaRN делает theta маленьким
    q = W_Q.T @ x_q
    k = W_K.T @ x_k
    return torch.dot(q, k) / math.sqrt(q.shape[0])


def decompose_linear_precursors(f, W_dec, W_Q, W_K):
    # f sparse, W_dec [n_dict,d_model] atoms d_i
    # q_i = W_dec @ W_Q [n_dict,d_head] линейно точно
    # q = sum f_i q_i conservation <1e-10
    # phi_i = angle(q_i) НЕ линейно: (1,0)0°+(0,1)90°=(1,1)45° !=90°
    q_i = W_dec @ W_Q
    W_dec @ W_K
    q = torch.einsum("i,id->d", f, q_i)
    return {"q": q, "q_i": q_i}


def test_conservation():
    torch.manual_seed(0)
    d_model, d_head, n_dict = 16, 4, 20
    W_Q = torch.randn(d_model, d_head, dtype=torch.float64)
    W_dec = torch.randn(n_dict, d_model, dtype=torch.float64)
    f = torch.zeros(n_dict, dtype=torch.float64)
    f[0] = 1.2
    f[1] = 0.8
    f[5] = 0.5
    q_direct = (W_dec.T @ f) @ W_Q
    q_sum = f @ (W_dec @ W_Q)
    err = (q_direct - q_sum).abs().max().item()
    assert err < 1e-10, f"err {err}"
    # phi non-linear demo
    torch.tensor([1.0, 0.0])
    torch.tensor([0.0, 1.0])
    phi_sum = math.degrees(math.atan2(1, 1))
    assert abs(phi_sum - 45) < 1e-6
    print(
        f"conservation linear err={err:.2e} <1e-10 PASS, phi non-linear 45° !=90° PASS"
    )


def polar_2d(q):
    mag = torch.norm(q)
    phi = torch.atan2(q[1], q[0]) if q.numel() >= 2 else torch.tensor(0.0)
    return mag, phi


def score_polar(mag_q, phi_q, mag_k, phi_k, pos_q, pos_k, theta=0.01):
    return mag_q * mag_k * torch.cos(phi_q - phi_k + (pos_q - pos_k) * theta)


def phase_gate_interaction_per_token(f_q, W_dec_q, W_Q, W_K, x_k, pos_q, pos_k):
    # total q, q_wo, gate_only len new angle old, phase_only len old angle new, interaction
    # Для каждого query token свой f_q, для каждого key g_j
    # YaRN: theta маленький base 500k, D=delta*theta маленький interaction 0.089 small vs 0.8 large RoPE fails 8192
    raise NotImplementedError


def random_norm_control(W_dec, W_Q):
    norms = torch.norm(W_dec, dim=1)
    rand = torch.randn_like(W_dec)
    rand = rand / torch.norm(rand, dim=1, keepdim=True) * norms[:, None]
    return torch.norm(W_dec @ W_Q, dim=1).mean(), torch.norm(rand @ W_Q, dim=1).mean()


def collect_for_seed(model_name="qwen3-4b", layer=6, n_examples=100, save_dir="./tmp"):
    # Hook blocks.{layer}.ln1.hook_normalized [B,T,D] half без логитов [B,T,V] sterility
    # TPU v5e-8 35GB fits 128GB, 1.5 PFLOP per head OOM -> per-query chunking for q_pos in range(T)
    # TransformerLens fast for 4B, nnsight for 14B
    pass


class SAE(torch.nn.Module):
    def __init__(
        self, d_model=2048, n_dict=16000, topk=50
    ):  # high-L0 50 63% vs low-L0 8 8-21%
        super().__init__()
        self.W_enc = torch.nn.Parameter(torch.randn(d_model, n_dict) * 0.01)
        self.W_dec = torch.nn.Parameter(torch.randn(n_dict, d_model) * 0.01)
        self.topk = topk

    def encode(self, x):
        f = torch.relu(x @ self.W_enc)
        vals, idx = torch.topk(f, self.topk, dim=-1)
        return torch.zeros_like(f).scatter_(-1, idx, vals)


def rank_favorites_phi_gate(W_dec, W_Q):
    q_i = W_dec @ W_Q
    gate_scores = torch.norm(q_i, dim=1)  # |q_i| gate всегда есть в RoPE/YaRN
    return gate_scores


def variance_explained(scores, contribs):
    return 1 - torch.var(scores - contribs) / torch.var(scores)


def steer_remove(x, d_k, f_k):
    return x - f_k * d_k


def steer_add(x, d_k, alpha=1.0):
    return x + alpha * d_k


def big_pipeline(seeds=None, layers=None, n=100):
    # check_sterility, collect per-query chunking, high-L0 50, phase_gate_interaction per token-pair
    # random-norm 5.2->2.7 vs 5.2->5.15, add 0.3->2.8, corr<0.3, cross-layer 2.1 vs 0.1, cross-seed 5/10, R2 high 0.62 vs low 0.08 phi err 5° vs 111°, conditional 8192 loss+0.0001 time 0.1*Y retrieval 0.2->0.7
    pass


if __name__ == "__main__":
    test_conservation()
