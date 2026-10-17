"""Single-input port of microsoft/LoRA gpt2_beam.py candidate selection.

MIT source at c4593f060e6a368d7bb5af5273b8e42810cdef90. Uses HF KV cache,
but preserves generated-only ngrams, softmax/log, top-k and EOS scoring.
"""

import torch


def generate(model, input_ids, settings):
    if input_ids.shape[0] != 1:
        raise ValueError("Source LoRA beam expects one context")
    beams, limit = settings["num_beams"], settings["max_new_tokens"]
    eos, ngram = settings["eos_token_id"], settings["no_repeat_ngram_size"]
    eos = eos if isinstance(eos, list) else [eos]
    ids = input_ids.repeat(beams, 1)
    history = None
    cache = None
    scores = torch.zeros(beams, device=ids.device)
    best_score, best = None, None
    for step in range(limit):
        result = model(input_ids=ids, past_key_values=cache, use_cache=True)
        logits, cache = result.logits[:, -1, :], result.past_key_values
        if ngram and history is not None and step + 1 >= ngram:
            for i, tokens in enumerate(history.tolist()):
                prefix = tuple(tokens[-(ngram - 1) :]) if ngram > 1 else ()
                banned = [
                    tokens[j + ngram - 1]
                    for j in range(len(tokens) - ngram + 1)
                    if tuple(tokens[j : j + ngram - 1]) == prefix
                ]
                logits[i, banned] = -float("inf")
        logprob = torch.log(torch.softmax(logits, dim=-1))
        candidates = logprob[0] if step == 0 else (scores[:, None] + logprob).flatten()
        scores, flat = torch.topk(candidates, beams, largest=True, sorted=True)
        parent, tokens = flat // logits.shape[-1], flat % logits.shape[-1]
        if hasattr(cache, "reorder_cache"):
            cache.reorder_cache(parent)
        else:
            cache = tuple(tuple(t.index_select(0, parent) for t in layer) for layer in cache)
        ids = tokens[:, None]
        history = ids if history is None else torch.cat([history[parent], ids], dim=1)
        for i in range(beams):
            if tokens[i].item() in eos:
                value = scores[i] / history.shape[1] ** settings["length_penalty"]
                if best_score is None or value > best_score:
                    best_score, best = value.clone(), history[i].clone()
                scores[i] = -float("inf")
    for i in range(beams):
        value = scores[i] / limit ** settings["length_penalty"]
        if best_score is None or value > best_score:
            best_score, best = value.clone(), history[i].clone()
    return torch.cat([input_ids[0], best])[None, :]
