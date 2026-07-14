"""Neural Grep Transpiler CLI.

Translates a natural-language query into GrepParameters JSON via a
finetuned Needle checkpoint, then executes ripgrep with those parameters.

Usage:
    needle/.venv/bin/python cli.py "find async functions starting with fetch or pull"
    needle/.venv/bin/python cli.py --path src/ --dry-run "find TODO comments"
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

# See server.py for why: JAX preallocates most of the GPU by default, which
# a 26M-param model doesn't need. Must precede any jax import below.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "needle"))

from schema import TOOLS_JSON, validate_grep_params  # noqa: E402

_CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

_model_cache = {}

# Leading trigger phrases stripped from the query when falling back to a
# literal search (see query_to_grep_params). Mirrors the phrase templates
# in data_gen.py's LITERAL_PHRASES/FUZZY_PHRASES/REGEX_PHRASES pools -- if
# the model produced invalid output, quoting the *entire* raw sentence
# (including "find"/"search for") as one literal term is almost always
# useless; stripping the trigger phrase at least searches for the actual
# subject of the query.
_FILLER_PREFIX_RE = re.compile(
    r"^(please\s+)?(can you\s+)?"
    r"(find all instances of|find everything related to|find every occurrence of|"
    r"find all|find|search for|search the codebase for|search the repo for|"
    r"scan the repo for|look for|grep for|locate|pattern-match for|"
    r"where is|where do we handle|show me the|show me|point me to|"
    r"track down the|track down|surface the code for|"
    r"dig up the code responsible for|write a regex to find)\s+",
    re.IGNORECASE,
)


def _fallback_term(nl_query):
    """Strip a leading trigger phrase, if any, for use as a literal fallback term."""
    prev = None
    q = nl_query.strip()
    while prev != q:
        prev = q
        q = _FILLER_PREFIX_RE.sub("", q, count=1).strip()
    return q or nl_query


def default_checkpoint():
    candidates = sorted(glob.glob(os.path.join(_CHECKPOINT_DIR, "needle_finetuned_*_best.pkl")))
    if not candidates:
        raise FileNotFoundError(
            f"No finetuned checkpoint found in {_CHECKPOINT_DIR}. "
            f"Run `needle finetune` first."
        )
    return candidates[-1]


def _load_model(checkpoint_path):
    if checkpoint_path in _model_cache:
        return _model_cache[checkpoint_path]

    from needle import load_checkpoint, SimpleAttentionNetwork, get_tokenizer

    params, config = load_checkpoint(checkpoint_path)
    model = SimpleAttentionNetwork(config)
    tokenizer = get_tokenizer()
    _model_cache[checkpoint_path] = (model, params, tokenizer)
    return model, params, tokenizer


def query_to_grep_params(nl_query, checkpoint_path=None):
    """Translate a natural-language query into validated GrepParameters.

    Returns (params_dict_or_None, outcome) where outcome is one of:
      - "matched":  model produced a valid ripgrep_search call; params is set.
      - "declined": model explicitly decided no tool applies (answers=[]);
                     params is None. This is the model doing the right thing
                     for out-of-domain queries (see the no_match training
                     category in data_gen.py), not a failure.
      - "fallback":  model output was malformed/invalid; params falls back to
                     a literal search of the raw query text.
    """
    from fast_generate import generate_fixed_shape

    checkpoint_path = checkpoint_path or default_checkpoint()
    model, params, tokenizer = _load_model(checkpoint_path)

    # Fixed-shape encoding avoids JAX recompiling the encoder graph for every
    # distinct query length (~5-6s per new length otherwise) -- see
    # fast_generate.py. Not relevant for a one-shot CLI process (recompile
    # happens once regardless), but keeps behavior identical if this is ever
    # called from a long-lived process handling many different queries.
    raw = generate_fixed_shape(model, params, tokenizer, nl_query, TOOLS_JSON, constrained=True)

    try:
        calls = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        calls = None

    if calls == []:
        return None, "declined"

    try:
        args = calls[0]["arguments"]
    except (TypeError, KeyError, IndexError):
        args = None

    if args is not None and validate_grep_params(args):
        return args, "matched"

    # Fallback: strip a leading trigger phrase ("find"/"search for"/etc.) and
    # use what's left as a single literal search term -- quoting the whole
    # raw sentence including the trigger phrase is almost never useful.
    return {"terms": [_fallback_term(nl_query)], "is_regex": False, "case_insensitive": False}, "fallback"


def build_rg_command(params, path="."):
    cmd = ["rg"]
    if params.get("case_insensitive"):
        cmd.append("-i")
    if not params.get("is_regex"):
        cmd.append("-F")
    for term in params["terms"]:
        cmd.extend(["-e", term])
    cmd.append(path)
    return cmd


def main():
    p = argparse.ArgumentParser(description="Natural language to ripgrep transpiler")
    p.add_argument("query", nargs="+", help="Natural language search query")
    p.add_argument("--path", default=".", help="Path to search (default: .)")
    p.add_argument("--checkpoint", default=None, help="Path to a finetuned .pkl checkpoint")
    p.add_argument("--dry-run", action="store_true", help="Print the rg command without running it")
    args = p.parse_args()

    nl_query = " ".join(args.query)
    params, outcome = query_to_grep_params(nl_query, checkpoint_path=args.checkpoint)

    if outcome == "declined":
        print(f"[ngt] not a search query -- no tool call generated for: {nl_query!r}", file=sys.stderr)
        return
    if outcome == "fallback":
        print(f"[ngt] model output failed validation; falling back to literal search for: {nl_query!r}", file=sys.stderr)

    cmd = build_rg_command(params, path=args.path)
    print(f"[ngt] {' '.join(cmd)}", file=sys.stderr)

    if args.dry_run:
        return
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
