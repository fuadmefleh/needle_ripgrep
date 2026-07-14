# loom-ripgrep

The ripgrep specialist in **Loom**, a federation of tiny (~26M-param)
single-tool models -- each finetuned for exactly one tool's grammar, meant
to eventually be coordinated by a router model that dispatches a single
user prompt to whichever specialist actually applies (see
[Loom](#loom) below). This repo is the first specialist: a model that
translates a natural-language query into a strict
`{terms, is_regex, case_insensitive}` JSON block, executed by [ripgrep](https://github.com/BurntSushi/ripgrep).
No embeddings, no vector index, no cloud call -- a tiny model parses intent,
`rg` searches the disk.

```
"find async functions starting with fetch or pull"
   -> {"terms": ["async def (fetch|pull)_"], "is_regex": true, "case_insensitive": false}
   -> rg -e "async def (fetch|pull)_" .
```

Built by finetuning [Needle](https://github.com/cactus-compute/needle)
(`Cactus-Compute/needle`, MIT), a 26M-parameter encoder-decoder model
purpose-built for single-shot function/tool calling, on a synthetic dataset
(7000 examples) covering three query styles:

- **literal** (40%) -- exact config keys / identifiers / markers, a broad
  pool of generic/exotic/foreign-loanword single words and multi-word
  phrases, non-software domain jargon (legal/medical/culinary/sports),
  and symbol-heavy literal targets (emails, shell commands, code syntax)
  so the model handles arbitrary text, not just code-shaped identifiers
- **fuzzy** (30%) -- abstract intent -> concrete keywords (e.g. "payment failures" -> `payment_failed`, `stripe_error`, `PaymentException`), across 87 concepts spanning backend/frontend/mobile/ML/infra/gaming/security
- **regex** (30%) -- structural pattern descriptions -> regex, 35 pattern types (emails, UUIDs, dates, hex/binary literals, markdown headers, etc.)

Phrasing includes typo-tolerant/terse variants ("fnd X", "can u find X")
across all three categories.

On a 565-example held-out set: 100% schema validity, 99.8% exact match,
100% regex-compile rate.

## Quickstart

```bash
git clone --recurse-submodules git@github.com:fuadmefleh/loom-ripgrep.git
cd loom-ripgrep

# 1. Set up Needle's own isolated venv (auto-detects GPU/CPU/TPU)
cd needle && source ./setup && cd ..

# 2. Put `ngt` and `ngt-server` on your PATH (symlinks into ~/.local/bin by default)
./install.sh

# 3. Start the warm server and query it
ngt-server start
ngt "find TODO comments"
ngt --path src/ "find code handling authentication errors"
```

A checkpoint (`checkpoints/needle_finetuned_*_best.pkl`) is committed
directly in this repo, so step 3 works immediately without training
anything.

## The `ngt` / `ngt-server` commands

`ngt-server` manages the persistent warm process:

```bash
ngt-server start     # loads the checkpoint, ~12.5s, prints "ready" when done
ngt-server status     # "running (pid N)" or "not running"
ngt-server stop        # stops it
ngt-server restart      # stop + start
```

`ngt` is the actual query command. It auto-detects whether `ngt-server` is
running (checks for its Unix socket) and uses it for fast (~500ms) queries;
if the server isn't running, it transparently falls back to a standalone
one-shot process (~12.5s cold start per call, no server needed):

```bash
ngt "find TODO comments"
ngt --path some/dir "find email addresses"
ngt --dry-run "find async functions starting with fetch or pull"   # print the rg command without running it
```

Both are plain shell scripts (`bin/ngt`, `bin/ngt-server`) that shell out to
`needle/.venv/bin/python` -- `install.sh` just symlinks them onto your PATH,
there's no packaging/build step. Run `install.sh <dir>` to link somewhere
other than `~/.local/bin`.

## Architecture

```
loom-ripgrep/
  needle/              git submodule: cactus-compute/needle (own .venv, never committed)
  bin/ngt, bin/ngt-server  the installable CLI commands (shell wrappers)
  install.sh            symlinks bin/ngt(-server) onto your PATH
  schema.py            the GrepParameters tool schema, shared by every script below
  data_gen.py           deterministic synthetic data generator (literal/fuzzy/regex)
  fast_generate.py       shape-bucketed inference (avoids per-query JIT recompiles)
  server.py / client.py   persistent warm process + Unix-socket client (~500ms/query) -- what bin/ngt(-server) drive
  cli.py                 standalone one-shot CLI (no server needed, ~12s cold start/call) -- bin/ngt's fallback
  eval_grep.py            held-out evaluation: schema validity, exact match, regex-compile rate
  benchmark.py            steady-state latency/throughput benchmark
  checkpoints/            finetuned model weights
  data/                   the exact train/eval JSONL this checkpoint was trained on
```

### Why a persistent server?

Each CLI invocation is a fresh Python process, so it always pays a ~12.5s
cold start (checkpoint load + JAX JIT compile). `server.py` loads the model
once and keeps it warm; `client.py` talks to it over a Unix socket, giving
~500-600ms per query regardless of query length. That consistency required
fixing a real gotcha in Needle's own `generate()`: it doesn't pad the
encoder input to a fixed shape, so JAX recompiles the whole encoder graph
for every distinct query length (~5-6s per new length). `fast_generate.py`
pads every query into one fixed token bucket instead, so there's exactly
one compile for the whole process lifetime.

### Behavior on out-of-domain input

This model always attempts to extract *some* search terms, for any input --
including queries that aren't search requests at all (task instructions,
questions, chit-chat). It does not refuse or return an empty result; it
does its best-effort extraction and lets `rg` return zero matches if nothing
fits. Deciding *whether* a query belongs to this tool at all is a router's
job, not this specialist's -- `data_gen.py` includes an unused
`gen_no_match` generator (answers=`[]`) kept specifically for training a
future router model that dispatches between multiple single-tool specialists
like this one.

## Regenerating the dataset / retraining

```bash
python3 data_gen.py --count 7000 --seed-start 0 --output data/ngt_data.jsonl
python3 data_gen.py --count 700 --seed-start 1000000 \
  --output data/ngt_eval.jsonl --exclude-file data/ngt_data.jsonl

./needle/.venv/bin/needle finetune data/ngt_data.jsonl \
  --checkpoint-dir checkpoints --epochs 30

./needle/.venv/bin/python eval_grep.py
```

`needle finetune` always re-downloads the pretrained base checkpoint from
`Cactus-Compute/needle` on Hugging Face and finetunes fresh, so reruns don't
compound on top of a previous finetune.

### GPU memory

JAX preallocates most of the GPU by default (and Needle's own `setup`
script pushes that to 95% via `XLA_PYTHON_CLIENT_MEM_FRACTION`, a setting
meant for large training jobs). Every entry-point script here
(`server.py`, `cli.py`, `benchmark.py`, `eval_grep.py`) sets
`XLA_PYTHON_CLIENT_PREALLOCATE=false` before importing JAX, which brings
actual usage down to ~700MB for this 26M-param model -- important if you're
sharing the GPU with anything else. `needle finetune` itself is
unaffected (training legitimately benefits from a larger arena) and will
still preallocate normally; stop `ngt-server` before retraining so it
isn't competing for GPU memory.

## Loom

The idea behind Loom: instead of one model juggling many tool schemas (which
dilutes accuracy per tool -- getting this one specialist's fuzzy-intent
category right already took real tuning), keep every tool as its own
narrowly-finetuned ~26M-param Needle checkpoint with maximally tight
grammar-constrained decoding, and put a separate router model in front that
picks which specialist a given prompt should go to -- using Needle's own
contrastive retrieval head (`retrieve_tools`/`encode_for_retrieval`) rather
than a full generate() call, so routing is cheap (one encoder pass) and
naturally supports "none of these apply" via a low similarity score.

This repo is specialist #1. The router and future specialists don't exist
yet -- `data_gen.py`'s unused `gen_no_match` generator (answers=`[]`, see
[Behavior on out-of-domain input](#behavior-on-out-of-domain-input) above)
is training-data groundwork for that router, kept here rather than built
into this specialist per the tradeoff above.

## License

MIT -- see [LICENSE](LICENSE). Needle itself (vendored as a submodule) is
also MIT, see [needle/LICENSE](https://github.com/cactus-compute/needle/blob/main/LICENSE).
