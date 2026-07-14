"""Held-out evaluation for the Neural Grep Transpiler checkpoint.

Regenerates the same held-out set that ngt_eval.jsonl was written from
(deterministic given the same count/seed/exclude args used by data_gen.py)
so per-example category labels are available without adding an extra field
to the on-disk training/eval JSONL files.

Reports, per search category (literal / fuzzy / regex):
  - schema validity rate  (parses as JSON and matches the GrepParameters schema)
  - term-set match rate   (terms as a set, is_regex, case_insensitive all match)
  - exact match rate      (terms list, in order, is_regex, case_insensitive all match)
  - regex-compile rate    (regex category only: predicted terms compile as regex)

And separately for no_match (out-of-domain queries, ground truth answers=[]):
  - decline rate          (model correctly output [] -- no tool call)
  - false-trigger rate    (model wrongly emitted a ripgrep_search call anyway)
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "needle"))

import data_gen  # noqa: E402
from schema import validate_grep_params  # noqa: E402

_CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def default_checkpoint():
    candidates = sorted(glob.glob(os.path.join(_CHECKPOINT_DIR, "needle_finetuned_*_best.pkl")))
    if not candidates:
        raise FileNotFoundError(f"No finetuned checkpoint found in {_CHECKPOINT_DIR}.")
    return candidates[-1]


def load_held_out_set(train_path, eval_count, eval_seed_start):
    train_queries = set()
    with open(train_path) as f:
        for line in f:
            if line.strip():
                train_queries.add(json.loads(line)["query"])
    rows, _counts = data_gen.generate(eval_count, eval_seed_start, exclude=train_queries)
    return rows


def score(pred_args, ref_args):
    """Returns (valid, term_set_match, exact_match)."""
    valid = validate_grep_params(pred_args) if isinstance(pred_args, dict) else False
    if not valid:
        return False, False, False

    term_set_match = (
        set(pred_args["terms"]) == set(ref_args["terms"])
        and pred_args["is_regex"] == ref_args["is_regex"]
        and pred_args["case_insensitive"] == ref_args["case_insensitive"]
    )
    exact_match = (
        pred_args["terms"] == ref_args["terms"]
        and pred_args["is_regex"] == ref_args["is_regex"]
        and pred_args["case_insensitive"] == ref_args["case_insensitive"]
    )
    return valid, term_set_match, exact_match


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--train-file", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ngt_data.jsonl"))
    p.add_argument("--eval-count", type=int, default=300)
    p.add_argument("--eval-seed-start", type=int, default=1000000)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    checkpoint_path = args.checkpoint or default_checkpoint()
    print(f"Loading checkpoint: {checkpoint_path}")

    from needle import load_checkpoint, SimpleAttentionNetwork, get_tokenizer, generate_batch

    params, config = load_checkpoint(checkpoint_path)
    model = SimpleAttentionNetwork(config)
    tokenizer = get_tokenizer()

    rows = load_held_out_set(args.train_file, args.eval_count, args.eval_seed_start)
    print(f"Evaluating on {len(rows)} held-out examples")

    stats = defaultdict(lambda: {"n": 0, "valid": 0, "term_set": 0, "exact": 0,
                                  "regex_n": 0, "regex_compiles": 0})
    no_match_stats = {"n": 0, "declined": 0, "false_trigger": 0, "malformed": 0}

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        preds = generate_batch(
            model, params, tokenizer,
            [r["query"] for r in batch],
            [r["tools"] for r in batch],
        )
        for row, pred_text in zip(batch, preds):
            category = row["category"]
            ref_calls = json.loads(row["answers"])

            try:
                pred_calls = json.loads(pred_text)
            except (json.JSONDecodeError, TypeError):
                pred_calls = None

            if category == "no_match":
                no_match_stats["n"] += 1
                if pred_calls == []:
                    no_match_stats["declined"] += 1
                elif isinstance(pred_calls, list) and pred_calls and isinstance(pred_calls[0], dict) \
                        and pred_calls[0].get("name") == "ripgrep_search":
                    no_match_stats["false_trigger"] += 1
                else:
                    no_match_stats["malformed"] += 1
                continue

            ref_args = ref_calls[0]["arguments"]
            try:
                pred_args = pred_calls[0]["arguments"]
            except (TypeError, KeyError, IndexError):
                pred_args = None

            valid, term_set_match, exact_match = score(pred_args, ref_args)

            s = stats[category]
            s["n"] += 1
            s["valid"] += int(valid)
            s["term_set"] += int(term_set_match)
            s["exact"] += int(exact_match)
            if category == "regex":
                s["regex_n"] += 1
                if valid and all(_compiles(t) for t in pred_args["terms"]):
                    s["regex_compiles"] += 1

    total = {"n": 0, "valid": 0, "term_set": 0, "exact": 0, "regex_n": 0, "regex_compiles": 0}
    print(f"\n{'category':<10} {'n':>5} {'valid%':>8} {'term_set%':>10} {'exact%':>8} {'regex_ok%':>10}")
    for category in sorted(stats):
        s = stats[category]
        for k in total:
            total[k] += s[k]
        regex_ok = f"{100 * s['regex_compiles'] / s['regex_n']:.1f}" if s["regex_n"] else "-"
        print(f"{category:<10} {s['n']:>5} {100*s['valid']/s['n']:>7.1f}% {100*s['term_set']/s['n']:>9.1f}% "
              f"{100*s['exact']/s['n']:>7.1f}% {regex_ok:>10}")

    regex_ok_total = f"{100 * total['regex_compiles'] / total['regex_n']:.1f}" if total["regex_n"] else "-"
    print(f"{'TOTAL':<10} {total['n']:>5} {100*total['valid']/total['n']:>7.1f}% {100*total['term_set']/total['n']:>9.1f}% "
          f"{100*total['exact']/total['n']:>7.1f}% {regex_ok_total:>10}")

    if no_match_stats["n"]:
        n = no_match_stats["n"]
        print(f"\n{'no_match':<10} {n:>5} {'declined%':>10} {'false_trig%':>12} {'malformed%':>11}")
        print(f"{'':<10} {'':>5} {100*no_match_stats['declined']/n:>9.1f}% "
              f"{100*no_match_stats['false_trigger']/n:>11.1f}% {100*no_match_stats['malformed']/n:>10.1f}%")


def _compiles(term):
    try:
        re.compile(term)
        return True
    except re.error:
        return False


if __name__ == "__main__":
    main()
