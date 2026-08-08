#!/usr/bin/env bash
# run_all.sh — MeshingLLM ICEBE2026 审稿修改版实验一键复现
set -e
cd "$(dirname "$0")"

echo "================================================"
echo "  MeshingLLM · ICEBE 2026 审稿版实验"
echo "  数据集: FinanceBench + MMLU-Accounting + Amazon + SHOPPING Intent"
echo "  基线: GPTQ + LoRA-r8 + FullFT-approx + FouRA + SeLoRA + LoCA"
echo "  CR=参数压缩比（非推理加速，见论文§6.2(1))"
echo "================================================"

echo ""
echo "==== [1/5] 数据准备（下载4个数据集 + 切片）===="
python data_prep.py

echo ""
echo "==== [2/5] MeshingLLM 核心算法冒烟测试（含QR→mask验证）===="
python meshing.py

echo ""
echo "==== [3/5] 频域PEFT基线（FouRA/SeLoRA/LoCA — 审稿版新增）===="
python train_eval.py --mode spectral_peft

echo ""
echo "==== [4/5] 全部实验（传统基线 + MeshingLLM + 跨段 + 消融）===="
python train_eval.py --mode all

echo ""
echo "==== [5/5] 统计分析 ===="
python stats_analysis.py --results ../outputs/results.jsonl --metric accuracy --target "MeshingLLM (L2)"

echo ""
echo "================================================"
echo "  DONE！把 ../outputs/results.jsonl 里的真实数字"
echo "  填回论文 §5.2–§5.4 的表格即可投稿"
echo ""
echo "  注意：FouRA/SeLoRA/LoCA 是简化版 re-impl"
echo "  如将军找到官方 repo，优先用官方版替代"
echo "================================================"
