# MeshingLLM · ICEBE 2026 实验代码 README（审稿修改版）

## 目录
```
experiment_code/
├── config.py          # 集中配置（改这里）——含SHOPPING Intent+6基线
├── meshing.py         # 核心算法 Π_θ / Γ_D / Def 1-4 / QR→mask（审稿版修复）
├── data_prep.py       # 下载4个数据集 + 品类/地区/电商意图切片
├── train_eval.py      # 基线 + MeshingLLM 训练评估（含频域PEFT基线）
├── dp_variant.py      # DP变体（会议版不跑，保留备用）
├── stats_analysis.py  # 统计出表（mean±std / CI / Wilcoxon / Holm）
├── run_all.sh         # 一键复现（审稿版）
└── README.md
```

## 审稿版实验范围（vs 旧版）
|  | 旧版 | 审稿版 |
|---|---|---|
| 数据集 | 3 | **4** (+SHOPPING Intent) |
| 基线 | 3 | **6** (+FouRA/SeLoRA/LoCA) |
| 跨段 | 2对 | **3对** (+域间迁移) |
| 消融 | 2策略 | 2策略（不变） |
| 种子 | 2 | 2（不变） |

## ⚠️ 审稿版关键修改说明
1. **SHOPPING Intent**：真实电商意图分类（Google Shopping Queries Dataset），解决"电商名不副实"（审稿人1/2）
2. **FouRA/SeLoRA/LoCA**：频域PEFT直接竞品基线，解决"最相关baseline零对比"（审稿人2）——当前为简化版re-impl，如找到官方repo优先替换
3. **QR→mask**：`meshing.py`新增`qr_guided_mask()`，明确QR分解→DCT/DST系数索引的映射（审稿人3 P0）
4. **CR含义**：参数压缩比，非推理加速——代码注释和输出已标注（审稿人3 P0）

## 环境（Windows / Linux 通用）
```bash
# CPU版 PyTorch（将军电脑无NVIDIA显卡）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers>=4.44 datasets peft scipy pandas scikit-learn numpy accelerate
# auto-gptq 需要 CUDA，CPU方案用模拟量化（见 StepByStep.md）
```

## 复现
```bash
cd outputs/experiment_code
bash run_all.sh   # Linux — 一键复现全部实验
# 或逐步运行（Windows）
python data_prep.py
python meshing.py                           # 冒烟测试（含QR→mask验证）
python train_eval.py --mode spectral_peft   # 频域PEFT基线
python train_eval.py --mode all             # 全部实验
python stats_analysis.py --results ../outputs/results.jsonl
```

## 产出
- `../outputs/results.jsonl`：每行一个 {method, dataset, seed, accuracy, cr, vram, note}
- `../outputs/tables/`：统计表格（mean±std / 95% CI / 配对检验 / Holm校正）

## 详细步骤
请阅读 **StepByStep.md**——从环境部署到结果收集的每一步都写清楚了。
