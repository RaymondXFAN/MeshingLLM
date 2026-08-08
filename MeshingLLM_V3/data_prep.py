"""data_prep.py — 下载数据集并切片（审稿版：4个核心数据集）。

⚠️ 2026-07-31 更新：HF 数据集结构变动，适配实际可用源：
  1. FinanceBench 只有 train split（150条），无 test → 手动切分
  2. Amazon Reviews 原数据集 (keung2019multilingual) 已废弃
     → 改用 srvmishra832/multilingual-amazon-reviews-6-languages（1.2M条，6语）
  3. SHOPPING Intent 原数据集 (tasksource) 已不存在
     → 改用 jtlicardo/ecommerce-intent-6k（6000条电商意图分类）

切片产物落在 data/<segment>/ 下的 JSONL 文件，
供 train_eval.py 与跨段迁移实验直接使用。

运行前务必设置 HF 镜像（国内必须）：
  Windows: set HF_ENDPOINT=https://hf-mirror.com
  Linux/Mac: export HF_ENDPOINT=https://hf-mirror.com
"""
from __future__ import annotations
import json
from pathlib import Path
from datasets import load_dataset
from config import Config


def prep_financebench(cfg: Config):
    """下载 FinanceBench（公开版 ~150 条金融QA）。

    ⚠️ FinanceBench 只有 train split，没有 test。这里手动做 80/20 切分。
    论文标注 "FinanceBench (150 financial QA, public subset)"。
    """
    print("[financebench] 正在下载 ...")
    ds = load_dataset(cfg.financebench, split="train")
    print(f"[financebench] 原始条数: {len(ds)}")

    out = Path(cfg.data_root) / "financebench"
    out.mkdir(parents=True, exist_ok=True)

    # 手动 80/20 切分（因为 HF 上没有 test split）
    split = ds.train_test_split(test_size=0.2, seed=42)
    split["train"].to_json(str(out / "train.jsonl"))
    split["test"].to_json(str(out / "test.jsonl"))
    print(f"[financebench] train:{len(split['train'])} test:{len(split['test'])}")
    print(f"[financebench] -> {out}")


def prep_mmlu(cfg: Config):
    """下载 MMLU-Accounting 子集（正常，有 test split）。"""
    out = Path(cfg.data_root) / "mmlu_business"
    out.mkdir(parents=True, exist_ok=True)
    for sub in cfg.mmlu_business_subsets:
        print(f"[mmlu] 正在下载 {sub} ...")
        d = load_dataset("cais/mmlu", sub, split="test")
        d.to_json(str(out / f"{sub}.jsonl"))
        print(f"[mmlu] {sub} -> {out / (sub + '.jsonl')}  ({len(d)} rows)")


def slice_amazon(cfg: Config):
    """下载 Amazon Reviews 多语版，按品类和地区切片。

    数据源: srvmishra832/multilingual-amazon-reviews-6-languages
    该数据集 1.2M 条，6 种语言 (de/en/es/fr/ja/zh)，28 种品类。
    列名: review_id, product_id, reviewer_id, stars, review_body,
          review_title, language, product_category

    品类切片：Electronics vs Home_Improvement → 品类异构
    地区切片：en(欧美) vs zh(中文) → 地区异构
    """
    print("[amazon] 正在下载（1.2M条，可能需要几分钟）...")
    ds = load_dataset(cfg.amazon_reviews, split="train")
    print(f"[amazon] 原始条数: {len(ds)}")

    out = Path(cfg.data_root) / "amazon"
    out.mkdir(parents=True, exist_ok=True)

    for lang in ("en", "zh"):
        print(f"[amazon] 按 language={lang} 过滤 ...")
        lang_ds = ds.filter(lambda x: x["language"] == lang)
        if len(lang_ds) == 0:
            print(f"[amazon] {lang}: 0 rows, skip")
            continue

        # 保存完整语言数据
        lang_dir = out / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        lang_ds.to_json(str(lang_dir / "all.jsonl"))
        print(f"[amazon:{lang}] full -> {lang_dir / 'all.jsonl'}  ({len(lang_ds)} rows)")

        # 品类切片（只取 electronics 和 home_improvement）
        for cat in ("electronics", "home_improvement"):
            cat_ds = lang_ds.filter(lambda x: x["product_category"] == cat)
            if len(cat_ds) > 0:
                cat_dir = out / f"{lang}_{cat}"
                cat_dir.mkdir(parents=True, exist_ok=True)
                split = cat_ds.train_test_split(test_size=0.2, seed=42)
                split["train"].to_json(str(cat_dir / "train.jsonl"))
                split["test"].to_json(str(cat_dir / "test.jsonl"))
                print(f"[amazon:{lang}_{cat}] train:{len(split['train'])} test:{len(split['test'])}")
            else:
                print(f"[amazon:{lang}_{cat}] 0 rows, skip")


def prep_shopping(cfg: Config):
    """⭐ 下载电商意图分类数据集 (jtlicardo/ecommerce-intent-6k)。

    替代原 tasksource/Shopping-Queries-Dataset（已不存在）。
    该数据集 6000 条，列名: prompt, completion, language。
    取英文子集做意图分类评估。
    """
    print("[shopping] 正在下载 ...")
    out = Path(cfg.data_root) / "shopping_intent"
    out.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset(cfg.shopping_intent, split="train")
        print(f"[shopping] 原始条数: {len(ds)}, 列名: {ds.column_names}")

        # 取英文子集（language 字段可能是 "english"）
        en_ds = ds.filter(lambda x: x.get("language", "english").lower() == "english")
        if len(en_ds) == 0:
            # 如果 language 字段不存在或全为 english，直接用全数据
            en_ds = ds
            print("[shopping] 无 language 过滤，使用全数据")

        # 限制条数做会议版（≤5000）
        if len(en_ds) > 5000:
            en_ds = en_ds.shuffle(seed=42).select(range(5000))

        # 80/20 切分
        split = en_ds.train_test_split(test_size=0.2, seed=42)
        split["train"].to_json(str(out / "train.jsonl"))
        split["test"].to_json(str(out / "test.jsonl"))
        print(f"[shopping] train:{len(split['train'])} test:{len(split['test'])}")

    except Exception as e:
        print(f"[shopping] 下载失败: {e}")
        print("[shopping] 请手动搜索 'ecommerce intent' 数据集下载")
        # 创建空占位文件避免后续流程报错
        placeholder = [{"placeholder": True, "text": "手动填充"}]
        with open(out / "train.jsonl", "w") as f:
            for item in placeholder:
                f.write(json.dumps(item) + "\n")
        with open(out / "test.jsonl", "w") as f:
            for item in placeholder:
                f.write(json.dumps(item) + "\n")
        print(f"[shopping] 创建占位文件 → {out}")


def main():
    cfg = Config()
    print("=" * 50)
    print("MeshingLLM 数据准备（审稿版：4个数据集）")
    print("⚠️ 运行前务必设置 HF 镜像:")
    print("   Windows: set HF_ENDPOINT=https://hf-mirror.com")
    print("   Linux/Mac: export HF_ENDPOINT=https://hf-mirror.com")
    print("=" * 50)

    prep_financebench(cfg)
    prep_mmlu(cfg)
    slice_amazon(cfg)
    prep_shopping(cfg)

    print("\n" + "=" * 50)
    print("✅ 数据准备完成！")
    print("请检查 data/ 目录下各子目录是否有 JSONL 文件。")
    print("=" * 50)


if __name__ == "__main__":
    main()
