"""Shape-bucketed generation to avoid per-query JIT recompilation.

Needle's own `generate()` builds an encoder input array exactly as long as
`len(query_tokens) + 1 + len(tools_tokens)`, with no padding. Since JAX
compiles a fresh XLA graph for every distinct array shape it sees, and every
differently-sized query produces a differently-sized encoder input, calling
`generate()` with varying queries in a live process recompiles on nearly
every call (~5-6s each on this GPU) instead of reusing one compiled graph.

Fix: pad the encoder input up to a fixed bucket length with pad tokens
(exactly what `make_padding_mask` already exists to support) so every call
uses the same array shape. One compile per bucket, reused forever after.

The constant tool schema (TOOLS_JSON) is ~122 tokens; queries are short, so
one bucket safely covers real NGT queries.
"""

import sys

DEFAULT_BUCKET_LEN = 256


def generate_fixed_shape(model, params, tokenizer, query, tools, max_gen_len=None,
                          bucket_len=DEFAULT_BUCKET_LEN, constrained=True):
    import jax.numpy as jnp
    import numpy as np
    from needle.dataset.tokenizer import DEFAULT_MAX_GEN_LEN
    from needle.model.architecture import make_padding_mask
    from needle.model.run import _build_encoder_input, _get_decode_fn, normalize_tools, restore_tool_names

    max_gen_len = max_gen_len or DEFAULT_MAX_GEN_LEN
    name_map = {}
    tools, name_map = normalize_tools(tools)

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    enc_tokens = _build_encoder_input(tokenizer, query, tools, max_enc_len=bucket_len)
    if len(enc_tokens) > bucket_len:
        raise ValueError(
            f"query+tools ({len(enc_tokens)} tokens) exceeds bucket_len={bucket_len}; "
            f"pick a larger bucket_len"
        )
    enc_tokens = enc_tokens + [pad_id] * (bucket_len - len(enc_tokens))
    enc_input = jnp.array([enc_tokens])  # constant (1, bucket_len) shape across all calls

    src_mask = make_padding_mask(enc_input, pad_id)
    encoder_out, enc_mask = model.apply(
        {"params": params}, enc_input, src_mask=src_mask, method="encode"
    )

    dec_buffer = jnp.full((1, max_gen_len), pad_id, dtype=jnp.int32)
    dec_buffer = dec_buffer.at[0, 0].set(eos_id)
    decode_fn = _get_decode_fn(model, max_gen_len)

    constrained_decoder = None
    if constrained:
        from needle.model.constrained import build_constrained_decoder
        constrained_decoder = build_constrained_decoder([tools], tokenizer)

    generated_tokens = []
    logits = decode_fn(params, dec_buffer, encoder_out, enc_mask)
    for i in range(0, max_gen_len - 1):
        next_logits = logits[0, i]
        if constrained_decoder and constrained_decoder.is_active(0):
            logits_np = np.array(next_logits)
            logits_np = constrained_decoder.constrain_logits(logits_np, 0)
            next_token = int(np.argmax(logits_np))
        else:
            next_token = int(jnp.argmax(next_logits))
        if constrained_decoder:
            constrained_decoder.update(0, next_token)
        if next_token == eos_id:
            break
        generated_tokens.append(next_token)
        dec_buffer = dec_buffer.at[0, i + 1].set(next_token)
        logits = decode_fn(params, dec_buffer, encoder_out, enc_mask)

    result = tokenizer.decode(generated_tokens)
    if result.startswith("<tool_call>"):
        result = result[len("<tool_call>"):]
    if name_map:
        result = restore_tool_names(result, name_map)
    return result
