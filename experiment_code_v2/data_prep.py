"""data_prep.py — 下载数据集并切片（审稿版：4个核心数据集）。

切片产物落在 data/<segment>/ 下的 JSONL 文件，
供 train_eval.py 与跨段迁移实验直接使用。

数据集清单（审稿版）：
  1. FinanceBench       — GAP A 主域（commerce-adjacent）
  2. MMLU-Accounting    — 域内辅助（business reasoning）
  3. Amazon Reviews     — 跨段切片（品类+地区）
  4. SHOPPING Intent    — ⭐ 真实电商意图分类（审稿人要求补真实电商任务）
"""
from __future__ import annotations
import json
from pathlib import Path
from datasets import load_dataset
from config import Config


def prep_financebench(cfg: Config):
    """下载 FinanceBench（10,231 金融QA）。"""
    ds = load_dataset(cfg.financebench, split="test")
    out = Path(cfg.data_root) / "financebench"
    out.mkdir(parents=True, exist_ok=True)
    ds.to_json(str(out / "test.jsonl"))
    print(f"[financebench] -> {out / 'test.jsonl'}  ({len(ds)} rows)")


def prep_mmlu(cfg: Config):
    """下载 MMLU-Accounting 子集。"""
    out = Path(cfg.data_root) / "mmlu_business"
    out.mkdir(parents=True, exist_ok=True)
    for sub in cfg.mmlu_business_subsets:
        d = load_dataset("cais/mmlu", sub, split="test")
        d.to_json(str(out / f"{sub}.jsonl"))
        print(f"[mmlu] {sub} -> {out / (sub + '.jsonl')}  ({len(d)} rows)")


def slice_amazon(cfg: Config):
    """下载 Amazon Reviews 多语版，按品类和地区切片。

    品类切片：Electronics vs Home_Improvement -> 品类异构
    地区切片：en(欧美) vs zh(中文) -> 地区异构
    """
    out = Path(cfg.data_root) / "amazon"
    out.mkdir(parents=True, exist_ok=True)

    # 只下载我们需要的 2 种语言 × 品类切片
    for lang in ("en", "zh"):
        try:
            ds = load_dataset(cfg.amazon_reviews, lang, split="train")
        except Exception as e:
            print(f"[amazon] skip {lang}: {e}")
            # 备用镜像
            try:
                ds = load_dataset("amazon_reviews_multi", lang, split="train")
            except Exception as e2:
                print(f"[amazon] backup also failed for {lang}: {e2}")
                continue

        # 保存每个语言的完整数据
        lang_dir = out / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        ds.to_json(str(lang_dir / "all.jsonl"))
        print(f"[amazon:{lang}] full -> {lang_dir / 'all.jsonl'}  ({len(ds)} rows)")

        # 按品类切片（只取 Electronics 和 Home_Improvement）
        if "product_category" in ds.column_names:
            for cat in ("electronics", "home_improvement"):
                sub = ds.filter(lambda x: x.get("product_category", "").lower() == cat)
                if len(sub) > 0:
                    cat_dir = out / f"{lang}_{cat}"
                    cat_dir.mkdir(parents=True, exist_ok=True)
                    # 80/20 切分
                    split = sub.train_test_split(test_size=0.2, seed=42)
                    split["train"].to_json(str(cat_dir / "train.jsonl"))
                    split["test"].to_json(str(cat_dir / "test.jsonl"))
                    print(f"[amazon:{lang}_{cat}] -> train:{len(split['train'])} test:{len(split['test'])}")
        elif "category" in ds.column_names:
            for cat in ("Electronics", "Home_Improvement"):
                sub = ds.filter(lambda x: x.get("category", "") == cat)
                if len(sub) > 0:
                    cat_dir = out / f"{lang}_{cat}"
                    cat_dir.mkdir(parents=True, exist_ok=True)
                    split = sub.train_test_split(test_size=0.2, seed=42)
                    split["train"].to_json(str(cat_dir / "train.jsonl"))
                    split["test"].to_json(str(cat_dir / "test.jsonl"))
                    print(f"[amazon:{lang}_{cat}] -> train:{len(split['train'])} test:{len(split['test'])}")


def prep_shopping(cfg: Config):
    """⭐ 新增：下载 Google Shopping Queries Dataset（真实电商意图分类）。

    审稿人指出"电商名不副实"——这里补一个真实电商意图理解任务。
    数据集包含电商搜索query的意图标签（如exact/relevant/irrelevant等）。
    我们取其中适合LLM评估的子集做意图分类。

    HuggingFace: tasksource/Shopping-Queries-Dataset
    备用: 也可从 https://huggingface.co/datasets/GoogleResearch/shopping_queries 直接下载
    """
    out = Path(cfg.data_root) / "shopping_intent"
    out.mkdir(parents=True, exist_ok=True)

    try:
        # 主数据源
        ds = load_dataset(cfg.shopping_intent, split="train")
        # 切分（数据集可能很大，取前5000条做会议版）
        if len(ds) > 5000:
            ds = ds.shuffle(seed=42).select(range(5000))
        ds.to_json(str(out / "train.jsonl"))
        print(f"[shopping] -> {out / 'train.jsonl'}  ({len(ds)} rows)")
    except Exception as e:
        print(f"[shopping] primary source failed: {e}")
        # 备用数据源
        try:
            ds = load_dataset("GoogleResearch/shopping_queries", split="train")
            if len(ds) > 5000:
                ds = ds.shuffle(seed=42).select(range(5000))
            ds.to_json(str(out / "train.jsonl"))
            print(f"[shopping] backup -> {out / 'train.jsonl'}  ({len(ds)} rows)")
        except Exception as e2:
            print(f"[shopping] all sources failed: {e2}")
            print("[shopping] 请手动从 https://huggingface.co/datasets/GoogleResearch/shopping_queries 下载")
            # 创建空占位文件避免后续流程报错
            with open(out / "train.jsonl", "w") as f:
                json.dump({"placeholder": True}, f)
            print(f"[shopping] created placeholder at {out / 'train.jsonl'}")


def main():
    cfg = Config()
    print("=" * 50)
    print("MeshingLLM 数据准备（审稿版：4个数据集）")
    print("  1. FinanceBench       — GAP A 主域")
    print("  2. MMLU-Accounting    — 域内辅助")
    print("  3. Amazon Reviews     — 跨段切片")
    print("  4. SHOPPING Intent    — 真实电商意图（审稿人要求）")
    print("=" * 50)
    prep_financebench(cfg)
    prep_mmlu(cfg)
    slice_amazon(cfg)
    prep_shopping(cfg)
    print("\n[done] 数据准备完成。")


if __name__ == "__main__":
    main()
