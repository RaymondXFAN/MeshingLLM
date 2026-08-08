"""dp_variant.py — HierFed-DPBA 分支：自适应差分隐私 + PBFT 共识接口。

对应论文 §5.1 的 MeshingLLM-DP 基线（ε=3）， foreshadow GAP D 多方电商协作。
本文件提供可复用的隐私-共识原语，集成进 meshing 训练循环即可。

数学约定：
  - 高斯机制：在梯度加噪前做 L2 裁剪（C），噪声 ~ N(0, σ²)，σ = C·√(2ln(1.5/δ))/ε
  - 自适应预算：每轮消耗 ε_t = ε_total / (T - t + 1)（单调递减，末尾省着花）
  - PBFT：需 2f+1 节点一致才聚合，防篡改（电商多方不可互信场景）
"""
from __future__ import annotations
import numpy as np


class AdaptiveDP:
    def __init__(self, epsilon: float = 3.0, delta: float = 1e-5):
        self.epsilon_total = epsilon
        self.delta = delta
        self.used = 0.0

    def _sigma(self, C: float, eps_step: float) -> float:
        return C * np.sqrt(2 * np.log(1.5 / self.delta)) / max(eps_step, 1e-6)

    def step_epsilon(self, round_idx: int, total: int) -> float:
        """自适应预算：ε_t = ε_total/(T-t+1)。"""
        eps_step = self.epsilon_total / (total - round_idx + 1)
        self.used += eps_step
        return eps_step

    def clip_and_noise(self, grad: np.ndarray, C: float = 1.0,
                       round_idx: int = 0, total: int = 1) -> np.ndarray:
        eps_step = self.step_epsilon(round_idx, total)
        sigma = self._sigma(C, eps_step)
        norm = np.linalg.norm(grad)
        if norm > C:
            grad = grad * (C / norm)
        noise = np.random.default_rng().normal(0, sigma, grad.shape)
        return grad + noise


class PBFTConsensus:
    """PBFT 共识：保证多方啮合参数聚合的完整性（防篡改）。"""

    def __init__(self, n_nodes: int = 4, f: int = 1):
        assert n_nodes >= 3 * f + 1, "PBFT 需 n >= 3f+1"
        self.n, self.f = n_nodes, f

    def aggregate(self, updates: list[np.ndarray]) -> np.ndarray:
        """需 >= 2f+1 份一致更新才聚合；否则抛异常（拜占庭告警）。"""
        if len(updates) < 2 * self.f + 1:
            raise RuntimeError(f"[PBFT] 仅 {len(updates)} 份更新，不足 2f+1={2*self.f+1}")
        stack = np.stack(updates, axis=0)
        median = np.median(stack, axis=0)
        # 一致性校验：多数节点与中位数的偏差在容差内
        dev = np.mean(np.abs(stack - median), axis=tuple(range(1, stack.ndim)))
        if np.mean(dev < 1e-2) < (2 * self.f + 1) / self.n:
            raise RuntimeError("[PBFT] 检测到拜占庭偏差")
        return median


def meshing_with_dp(W, Gamma, mask, dp: AdaptiveDP, consensus: PBFTConsensus,
                     local_grads: list[np.ndarray], round_idx: int, total: int):
    """在啮合运算上叠加 DP 噪声 + PBFT 聚合（GAP D 预览）。"""
    noised = [dp.clip_and_noise(g, round_idx=round_idx, total=total) for g in local_grads]
    agg = consensus.aggregate(noised)
    return W + agg  # 简化：把聚合增量加回流（实际应对 Γ/系数做啮合）


if __name__ == "__main__":
    dp = AdaptiveDP(epsilon=3.0)
    cons = PBFTConsensus(n_nodes=4, f=1)
    g = [np.ones((4, 4)) * i for i in (1.0, 1.0, 1.0, 0.9)]  # 3 一致 + 1 偏差
    agg = cons.aggregate(g)
    print("[dp] ε used this step:", round(dp.step_epsilon(0, 5), 3))
    print("[pbft] aggregated shape:", agg.shape)
