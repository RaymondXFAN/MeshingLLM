#!/usr/bin/env bash
# ============================================================
# run_all_gpu.sh — MeshingLLM AutoDL 云GPU一键复现脚本
# ============================================================
# 适配配置：RTX 4090 (24GB) / CUDA 12.8 / PyTorch cu124
# 所有数据和项目必须放在 /root/autodl-tmp/ 下！
# ============================================================
# ⭐⭐⭐ 项目路径自动检测：
#   脚本会自动找到自己所在的目录作为项目根目录，
#   不管文件夹叫 MeshingLLM 还是 experiment_code 都能工作！
# ============================================================
set -e

# ---- ⭐ 自动检测项目路径（脚本所在目录 = 项目根目录） ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
DATA_DIR="${PROJECT_DIR}/data"
OUTPUT_DIR="${PROJECT_DIR}/outputs"

echo "================================================"
echo "  MeshingLLM · ICEBE 2026 · AutoDL GPU版实验"
echo "  基座: Qwen2.5-7B (FP16 on RTX 4090)"
echo "  数据集: FinanceBench + MMLU + Amazon + SHOPPING Intent"
echo "  基线: GPTQ + LoRA-r8 + FullFT + FouRA + SeLoRA + LoCA"
echo "  CR=参数压缩比（非推理加速，见论文§6.2(1))"
echo "  ⭐ 项目路径（自动检测）: ${PROJECT_DIR}"
echo "================================================"

# ---- Step 0: 环境准备 ----
echo ""
echo "==== [Step 0/7] 环境检查与准备 ===="

# 检查GPU
echo "[0] 检查GPU..."
nvidia-smi || { echo "❌ GPU不可用！请检查AutoDL实例是否正常"; exit 1; }
echo "✅ GPU可用"

# 设置HF镜像（国内必须！）
export HF_ENDPOINT=https://hf-mirror.com
echo "[0] HF镜像已设置: $HF_ENDPOINT"

# 设置GPU模式
export MESHINGLLM_GPU=1
echo "[0] MeshingLLM GPU模式已启用 (MESHINGLLM_GPU=1)"

# 创建数据和输出目录（如果不存在）
mkdir -p "${DATA_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/tables"

# ⭐ 检查代码是否存在（自动检测目录，不依赖文件夹名）
if [ ! -f "${PROJECT_DIR}/config.py" ]; then
    echo "❌ 代码文件不存在！config.py 不在 ${PROJECT_DIR}/"
    echo ""
    echo "⭐ 请确认："
    echo "   1. 代码zip已解压到某个目录"
    echo "   2. run_all_gpu.sh 和 config.py 在同一目录"
    echo "   3. 然后在那个目录运行 bash run_all_gpu.sh"
    echo ""
    echo "   查找代码位置: find /root/autodl-tmp -name 'config.py' 2>/dev/null"
    exit 1
fi
echo "✅ 代码已就绪: ${PROJECT_DIR}/"

# ---- Step 1: 安装依赖 ----
echo ""
echo "==== [Step 1/7] 安装Python依赖 ===="

# 先检查PyTorch是否已安装且支持CUDA
echo "[1] 检查PyTorch..."
python -c "import torch; print(f'PyTorch={torch.__version__}, CUDA={torch.cuda.is_available()}')" 2>/dev/null || {
    echo "[1] PyTorch未安装或不支持CUDA，正在安装PyTorch cu124..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
}

# 验证CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA不可用！'; print('✅ PyTorch CUDA可用')"
echo "✅ PyTorch环境正常"

# 安装其他依赖
echo "[1] 安装其他依赖..."
pip install transformers datasets peft scipy pandas scikit-learn numpy accelerate -q
echo "✅ 依赖安装完成"

# 尝试安装auto-gptq（INT4-GPTQ基线需要，失败则跳过用模拟量化）
echo "[1] 尝试安装auto-gptq（可选，失败不影响其他实验）..."
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu124/ 2>/dev/null || {
    echo "⚠️  auto-gptq安装失败，INT4-GPTQ基线将使用模拟量化（不影响其他实验）"
}

# ---- Step 2: 数据准备 ----
echo ""
echo "==== [Step 2/7] 下载4个数据集 + 切片 ===="
cd "${PROJECT_DIR}"
python data_prep.py
echo "✅ 数据准备完成"

# ---- Step 3: 核心算法冒烟测试 ----
echo ""
echo "==== [Step 3/7] MeshingLLM核心算法冒烟测试 ===="
cd "${PROJECT_DIR}"
python meshing.py
echo "✅ 冒烟测试通过"

# ---- Step 4: 频域PEFT基线 ----
echo ""
echo "==== [Step 4/7] 频域PEFT基线（FouRA/SeLoRA/LoCA） ===="
cd "${PROJECT_DIR}"
python train_eval.py --mode spectral_peft
echo "✅ 频域PEFT基线完成"

# ---- Step 5: 全部实验 ----
echo ""
echo "==== [Step 5/7] 全部实验（MeshingLLM + 基线 + 跨段 + 消融） ===="
echo "⏱️  这一步最耗时（7B模型加载+16层啮合），预计15-30分钟"
cd "${PROJECT_DIR}"
python train_eval.py --mode all
echo "✅ 全部实验完成"

# ---- Step 6: 统计分析 ----
echo ""
echo "==== [Step 6/7] 统计分析 ===="
cd "${PROJECT_DIR}"
python stats_analysis.py --results "${OUTPUT_DIR}/results.jsonl" --metric accuracy --target "MeshingLLM (L2)"
echo "✅ 统计分析完成"

# ---- Step 7: 结果汇总 ----
echo ""
echo "==== [Step 7/7] 结果汇总 ===="
echo ""
echo "================================================"
echo "  🎉 全部实验完成！"
echo ""
echo "  ⭐ 项目路径: ${PROJECT_DIR}"
echo ""
echo "  结果文件："
echo "    ${OUTPUT_DIR}/results.jsonl          — 实验原始数据"
echo "    ${OUTPUT_DIR}/tables/accuracy_table.md — 统计表格"
echo "    ${OUTPUT_DIR}/tables/accuracy_compare.csv — 对比检验"
echo ""
echo "  📋 下一步："
echo "    1. 查看结果: cat ${OUTPUT_DIR}/results.jsonl"
echo "    2. 把真实数字填入论文 §5.2–§5.4 表格"
echo "    3. 关机停计费: 在AutoDL控制台点击「关机」"
echo ""
echo "  ⚠️  关机前务必把 outputs/ 目录下载到本地！"
echo "     关机后数据盘(/root/autodl-tmp/)数据保留，"
echo "     但释放实例后数据会丢失！"
echo ""
echo "  注意：FouRA/SeLoRA/LoCA 是简化版 re-impl"
echo "  如将军找到官方repo，优先用官方版替代"
echo "================================================"
