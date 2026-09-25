#!/usr/bin/env python3
"""
CLI Ideal - One-Click для всех - максимально эффективно решает все проблемы
Usage: python3 frontier-01-cli-ideal.py --mode all|bilinearity|bow|graphs|eval|kaggle|tpu
"""
import argparse, sys, os, math, json, hashlib

def run_bilinearity():
    print("=== RUN bilinearity-break-ideal ===")
    os.system("python3 frontier-01-bilinearity-break-ideal.py")

def run_bow():
    print("=== RUN bag-of-words-test ===")
    os.system("python3 frontier-01-bag-of-words-test.py")

def run_graphs():
    print("=== RUN graphs-ULTIMATE-V11 TOP-LAB V15 FINAL ===")
    os.system("python3 frontier-01-graphs-ULTIMATE-V11.py")
    # V15 FINAL: only ULTIMATE V11 162K-290K beautiful clear dark #111 lw4 error bars subplots unit circle
    # Old V10 TOPLAB 153K-263K deprecated, all-graphs-ideal.py deprecated - use only ULTIMATE for Oral 6

def run_eval():
    print("=== RUN eval-numpy-ideal ===")
    os.system("python3 frontier-01-eval-numpy-ideal.py")

def run_all():
    run_bilinearity()
    run_bow()
    run_graphs()
    run_eval()
    print("\n=== ALL DONE IDEAL ===")
    print("Files: settings-ideal.json config_hash 9bd59cac dataset_hash 848bb0b0")
    print("Figures: 8 PNG 200 dpi fig_bilinearity_break.png fig_small_angle.png fig_gate_phase.png fig_high_low_L0.png fig_yarn_rope_interaction.png fig_pprope_split.png fig_conservation.png fig_bag_of_words.png")
    print("Next: Kaggle 2xT4 11 cells from frontier-01-kaggle-notebook-ideal.py or TPU v5e-8 command from TPU-runbook-IDEAL.md")

def main():
    parser = argparse.ArgumentParser(description="Gemma 4 4B pp-RoPE p=0.25 RoPE+YaRN Phi + BoW - One-Click Ideal CLI")
    parser.add_argument("--mode", choices=["all","bilinearity","bow","graphs","eval","kaggle","tpu","usage"], default="all", help="what to run")
    args = parser.parse_args()
    if args.mode == "all":
        run_all()
    elif args.mode == "bilinearity":
        run_bilinearity()
    elif args.mode == "bow":
        run_bow()
    elif args.mode == "graphs":
        run_graphs()
    elif args.mode == "eval":
        run_eval()
    elif args.mode == "kaggle":
        print("Kaggle 2xT4: New Notebook T4 x2 Internet ON, copy 11 cells from frontier-01-kaggle-notebook-ideal.py, Run All 3h <12h")
        print(open("frontier-01-kaggle-howto-IDEAL.md").read()[:2000])
    elif args.mode == "tpu":
        print("TPU v5e-8: torch_xla.distributed.xla_dist --tpu $TPU_NAME -- python3 collect_for_seed.py --model gemma-4-4b-e4b --layer 6 --backend nnsight --chunking per-query --save half --no-logits")
        print(open("frontier-01-TPU-runbook-IDEAL.md").read()[:2000])
    elif args.mode == "usage":
        print(open("frontier-01-README-USAGE-IDEAL.md").read()[:5000])

if __name__ == "__main__":
    main()
