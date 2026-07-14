# Neural Grep Transpiler (NGT)

A 26M-parameter model that translates a natural-language query into a strict
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
covering three query styles:

- **literal** (40%) -- exact config keys / identifiers / markers
- **fuzzy** (30%) -- abstract intent -> concrete keywords (e.g. "payment failures" -> `payment_failed`, `stripe_error`, `PaymentException`)
- **regex** (30%) -- structural pattern descriptions -> regex

On a 300-example held-out set: 100% schema validity, 96% exact match, 100%
regex-compile rate.

## Quickstart

```bash
git clone --recurse-submodules <this-repo-url>
cd neural_grep_transpiler

# 1. Set up Needle's own isolated venv (auto-detects GPU/CPU/TPU)
cd needle && source ./setup && cd ..

# 2. Run the persistent server (loads the checkpoint once, ~12s cold start)
./needle/.venv/bin/python server.py &

# 3. Query it
./needle/.venv/bin/python client.py "find TODO comments"
./needle/.venv/bin/python client.py --path src/ "find code handling authentication errors"
```

A checkpoint (`checkpoints/needle_finetuned_*_best.pkl`) is committed
directly in this repo, so step 2 works immediately without training
anything.

## Architecture

```
neural_grep_transpiler/
  needle/              git submodule: cactus-compute/needle (own .venv, never committed)
  schema.py            the GrepParameters tool schema, shared by every script below
  data_gen.py           deterministic synthetic data generator (literal/fuzzy/regex)
  fast_generate.py       shape-bucketed inference (avoids per-query JIT recompiles)
  server.py / client.py   persistent warm process + Unix-socket client (~500ms/query)
  cli.py                 standalone one-shot CLI (no server needed, ~12s cold start/call)
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
python3 data_gen.py --count 2400 --seed-start 0 --output data/ngt_data.jsonl
python3 data_gen.py --count 300 --seed-start 1000000 \
  --output data/ngt_eval.jsonl --exclude-file data/ngt_data.jsonl

./needle/.venv/bin/needle finetune data/ngt_data.jsonl \
  --checkpoint-dir checkpoints --epochs 30

./needle/.venv/bin/python eval_grep.py
```

`needle finetune` always re-downloads the pretrained base checkpoint from
`Cactus-Compute/needle` on Hugging Face and finetunes fresh, so reruns don't
compound on top of a previous finetune.

## License

MIT -- see [LICENSE](LICENSE). Needle itself (vendored as a submodule) is
also MIT, see [needle/LICENSE](https://github.com/cactus-compute/needle/blob/main/LICENSE).
