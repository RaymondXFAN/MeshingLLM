"""MeshingLLM 实验配置 — 审稿修改版（ICEBE 2026投稿）。

数据集 4个（+SHOPPING Intent）、基线 6个（+FouRA/SeLoRA/LoCA）。
改这里就行，别去动各模块里的硬编码～ (´• ω •`)

⭐ GPU模式切换：
  默认 = CPU模式（Qwen2.5-1.5B, device=cpu, batch=2）
  设置环境变量 MESHINGLLM_GPU=1 → GPU模式（Qwen2.5-7B, device=cuda, batch=4）
  在AutoDL上运行前：export MESHINGLLM_GPU=1
  在Windows CMD上运行前：set MESHINGLLM_GPU=1
"""
import os
from dataclasses import dataclass, field
from typing import List, Tuple

# ⭐ GPU模式检测：环境变量 MESHINGLLM_GPU=1 时切换到GPU配置
GPU_MODE = os.environ.get("MESHINGLLM_GPU", "0") == "1"


@dataclass
class Config:
    # ---- 基座模型 ----
    # ⭐ GPU模式：7B（论文正式数据）；CPU模式：1.5B（本地验证）
    base_model: str = "Qwen/Qwen2.5-7B" if GPU_MODE else "Qwen/Qwen2.5-1.5B"
    max_seq_len: int = 2048
    # ⭐ GPU模式：cuda；CPU模式：cpu
    device: str = "cuda" if GPU_MODE else "cpu"
    # ⭐ GPU模式：FP16加速；CPU模式：FP32
    torch_dtype: str = "float16" if GPU_MODE else "float32"

    # ---- 数据集（4个：审稿版新增SHOPPING Intent）----
    data_root: str = "data"
    financebench: str = "PatronusAI/financebench"                       # GAP A 主域（commerce-adjacent）
    mmlu_business_subsets: Tuple[str, ...] = (
        "professional_accounting",                                      # 只取1个子集，域内辅助
    )
    amazon_reviews: str = "srvmishra832/multilingual-amazon-reviews-6-languages"  # 跨段切片（品类+地区）[原keung2019已废弃]
    shopping_intent: str = "jtlicardo/ecommerce-intent-6k"            # ⭐ 电商意图分类（原tasksource已废弃）
    # ---- 以下为SCI版保留，会议版不跑 ----
    # banking77: str = "legacy-datasets/banking77"
    # clinc150: str = "PolyAI-NLP/clinc_oos"

    # ---- 啮合参数 ----
    cr_target: int = 16                  # 目标压缩比（§3.4 保守 CR=16）——注意：参数压缩，非推理加速
    tau_energy: float = 0.85             # §3.5 能量阈值 τ（AND交集策略下τ=0.85→CR≈10× on 1.5B；7B预期≈16×）
    k_clusters: int = 4                  # 三角频谱聚类 K（elbow）
    freq_strategy: str = "energy+gradient"   # 会议版只跑 energy+gradient 和 energy-only

    # ---- 三级渐进推理 ----
    level: str = "L2"                    # L1 / L2 / L3

    # ---- GPU专属参数 ----
    # ⭐ GPU模式：处理更多层（7B模型28层 vs 1.5B模型28层但每层更大）
    max_mesh_layers: int = 16 if GPU_MODE else 8   # GPU处理16层，CPU只处理8层（控制计算量）
    # ⭐ GPU模式：矩阵子块上限（7B权重矩阵更大）
    max_matrix_dim: int = 4096 if GPU_MODE else 2048   # GPU可处理4096子块，CPU只2048

    # ---- 训练 ----
    lr: float = 1e-4
    epochs: int = 3
    # ⭐ GPU模式：batch=4（RTX4090 24GB够用）；CPU模式：batch=2
    batch_size: int = 4 if GPU_MODE else 2
    seeds: Tuple[int, ...] = (42, 123)   # 会议版2种子足够，SCI加更多

    # ---- Spectral PEFT 基线说明 ----
    # FouRA/SeLoRA/LoCA 三个基线采用 "re-implementation following published descriptions"
    # （官方repo在投稿时未公开/不稳定），详见 train_eval.py 注释
    # 如将军能找到官方repo，优先用官方版替代此简化实现

    # ---- 输出 ----
    out_root: str = "outputs"

    def __post_init__(self):
        """打印当前配置摘要，方便确认是CPU还是GPU模式。"""
        mode = "GPU⚡" if GPU_MODE else "CPU🐢"
        print(f"[config] 模式={mode}, base_model={self.base_model}, device={self.device}, "
              f"batch_size={self.batch_size}, max_mesh_layers={self.max_mesh_layers}, "
              f"max_matrix_dim={self.max_matrix_dim}")

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
