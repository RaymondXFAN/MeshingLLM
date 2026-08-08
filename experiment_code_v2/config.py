"""MeshingLLM 实验配置 — 审稿修改版（ICEBE 2026投稿）。

数据集 4个（+SHOPPING Intent）、基线 6个（+FouRA/SeLoRA/LoCA）。
改这里就行，别去动各模块里的硬编码～ (´• ω •`)
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # ---- 基座模型 ----
    base_model: str = "Qwen/Qwen2.5-7B"
    max_seq_len: int = 2048
    device: str = "cpu"               # 无独立显卡，纯 CPU 方案；云 GPU 时改 "cuda"

    # ---- 数据集（4个：审稿版新增SHOPPING Intent）----
    data_root: str = "data"
    financebench: str = "PatronusAI/financebench"                       # GAP A 主域（commerce-adjacent）
    mmlu_business_subsets: Tuple[str, ...] = (
        "professional_accounting",                                      # 只取1个子集，域内辅助
    )
    amazon_reviews: str = "keung2019multilingual"                     # 跨段切片（品类+地区）
    shopping_intent: str = "tasksource/Shopping-Queries-Dataset"       # ⭐ 新增：真实电商意图分类（审稿人要求）
    # ---- 以下为SCI版保留，会议版不跑 ----
    # banking77: str = "legacy-datasets/banking77"
    # clinc150: str = "PolyAI-NLP/clinc_oos"

    # ---- 啮合参数 ----
    cr_target: int = 16                  # 目标压缩比（§3.4 保守 CR=16）——注意：参数压缩，非推理加速
    tau_energy: float = 0.95             # §3.5 能量阈值 τ
    k_clusters: int = 4                  # 三角频谱聚类 K（elbow）
    freq_strategy: str = "energy+gradient"   # 会议版只跑 energy+gradient 和 energy-only

    # ---- 三级渐进推理 ----
    level: str = "L2"                    # L1 / L2 / L3

    # ---- 训练 ----
    lr: float = 1e-4
    epochs: int = 3
    batch_size: int = 2                  # CPU方案 batch=2，云GPU可改4
    seeds: Tuple[int, ...] = (42, 123)   # 会议版2种子足够，SCI加更多

    # ---- Spectral PEFT 基线说明 ----
    # FouRA/SeLoRA/LoCA 三个基线采用 "re-implementation following published descriptions"
    # （官方repo在投稿时未公开/不稳定），详见 train_eval.py 注释
    # 如将军能找到官方repo，优先用官方版替代此简化实现

    # ---- 输出 ----
    out_root: str = "outputs"

    def segment_pairs(self) -> List[Tuple[str, str]]:
        """跨段迁移 — 审稿版：3对（品类+地区+域间），体现 Def 4 同质异构。"""
        return [
            ("amazon_electronics", "amazon_home"),     # 品类异构
            ("amazon_cn", "amazon_eu"),                # 地区异构
            ("financebench", "shopping_intent"),       # ⭐ 新增：域间迁移（金融→电商意图）
        ]

    def baselines(self) -> List[str]:
        """审稿版基线：6个（3传统 + 3频域PEFT直接竞品）。"""
        return ["INT4-GPTQ", "LoRA-r8", "FullFT-approx",
                "FouRA", "SeLoRA", "LoCA"]
