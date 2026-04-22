---
agent: agent
description: "Implement repo-specific OpenPI changes to decode Gemma hidden states to text and expose prefix text in policy outputs."
---

# Hidden State To Text Steps (OpenPI)

You are assisting with OpenPI code. Produce practical, repo-specific steps and code edits.

## Context

- You are a coding agent in this workspace.
- Prefer concrete code edits over abstract explanations when requests are actionable.
- If a requested file change is clear, implement it directly and keep edits minimal.
- Preserve existing APIs and style unless the user asks for a refactor.

## Task

Implement support for converting Gemma hidden states into readable text, and include prefix text derived from Pi0 hidden states in policy inference outputs.

## Primary Files

- `src/openpi/models/pi0.py`
- `src/openpi/policies/policy.py`
- `src/openpi/models/gemma.py`
- `src/openpi/models/tokenizer.py`

## Required Behavior

1. In `src/openpi/models/pi0.py`, determine which expert output to use from the first return value of `self.PaliGemma.llm(...)`:
- `[prefix_tokens, None]` -> expert index `0`
- `[None, suffix_tokens]` -> expert index `1`
- `[prefix_tokens, suffix_tokens]` -> both indices `0` and `1` exist

2. In `Pi0.sample_actions`, capture the currently ignored prefix hidden state at the prefix KV-cache call site.
- Replace `_` in `_, kv_cache = self.PaliGemma.llm([prefix_tokens, None], ...)` with a named variable.
- Use that prefix expert hidden state (index `0`) as the source for reconstructed prefix text.

3. Treat hidden states as continuous features, not token IDs.
- Hidden states have shape `[B, T, D]`.
- They must be projected to vocab logits `[B, T, V]` with Gemma decode weights.

4. Project hidden states to logits.
- Use `Embedder.decode` behavior (`x @ embedding_table.T`) from `src/openpi/models/gemma.py`.

5. Convert logits to token IDs.
- Deterministic: `argmax` on the vocabulary axis.
- Optional: mention sampling if requested.

6. Decode token IDs to text.
- Use PaliGemma SentencePiece tokenizer flow in `src/openpi/models/tokenizer.py`.
- Decode each batch element from token ID list to a string.

7. Fix the image-token segment issue before decoding.
- Do not decode the full prefix hidden sequence from `embed_prefix`; it includes image tokens first.
- Isolate only prompt-token hidden states using prompt length:
    - `prompt_len = observation.tokenized_prompt.shape[1]`
    - `prompt_hidden = hidden[:, -prompt_len:, :]`
- Use `observation.tokenized_prompt_mask` as the decoding mask for this segment.

8. Include masking guidance.
- If a token mask is available, remove padded positions before decoding.
- Warn that decoding padded positions adds junk text.

9. Add policy output support in `src/openpi/policies/policy.py`.
- Use prefix text produced from Pi0 hidden-state decoding, not `obs["prompt"]`.
- Return `prefix_text` as a Python string from the jitted model path.
- Add `prefix_text` to the policy output dict and preserve it in final responses.

10. Include validation checks.
- Selected expert output is not `None`.
- dtype and shapes are consistent.
- token IDs are in valid vocabulary range.
- Prompt segment shape matches mask shape: `prompt_hidden.shape[1] == tokenized_prompt_mask.shape[1]`.

11. Include limitations.
- Text from hidden-state reconstruction may not exactly match the original prompt.
- This is model-decoded text, not ground-truth input serialization.

## Output Format

Return all of the following sections in this order:

1. `Summary`
2. `Step-by-step`
3. `Minimal repo-specific code`
4. `Common failure modes`

## Minimal Code Template To Adapt (Pi0)

```python
# 1) Capture model outputs instead of ignoring them
(prefix_outputs_per_expert, kv_cache) = self.PaliGemma.llm(
    [prefix_tokens, None],
    mask=prefix_attn_mask,
    positions=positions,
)

# 2) Pick the desired expert output hidden states
hidden = prefix_outputs_per_expert[0]  # [B, T, D] for prefix-only call (previously ignored as `_`)
if hidden is None:
    raise ValueError("Selected expert output is None")

# 2.1) Isolate prompt-token hidden states (exclude image tokens at the front)
if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
    raise ValueError("tokenized_prompt and tokenized_prompt_mask are required to decode prefix text")
prompt_len = observation.tokenized_prompt.shape[1]
prompt_hidden = hidden[:, -prompt_len:, :]
token_mask = observation.tokenized_prompt_mask
if prompt_hidden.shape[1] != token_mask.shape[1]:
    raise ValueError("Prompt hidden length and token mask length do not match")

# 3) Project hidden states to vocab logits [B, T, V]
# Equivalent to Embedder.decode: logits = hidden @ embedding_table.T
embedder = self.PaliGemma.llm.embedder
logits = embedder(prompt_hidden, method="decode")

# 4) Convert logits to token IDs [B, T]
token_ids = jnp.argmax(logits, axis=-1)

# 5) Decode token IDs to text and return Python strings from sample_actions
from openpi.models import tokenizer as _tokenizer

token_ids_np = np.asarray(jax.device_get(token_ids))
tokenizer = _tokenizer.PaligemmaTokenizer(max_len=token_ids_np.shape[1])

texts = []
for b in range(token_ids_np.shape[0]):
    ids = token_ids_np[b].tolist()

    # Optional: if you have a token mask, drop padded positions before decode.
    if token_mask is not None:
        mask_b = np.asarray(jax.device_get(token_mask[b])).astype(bool).tolist()
        ids = [tok for tok, keep in zip(ids, mask_b) if keep]

    texts.append(tokenizer._tokenizer.decode(ids))

# 6) Return actions plus decoded prefix text string(s)
return {
    "actions": x_0,
    "prefix_text": texts[0] if len(texts) == 1 else texts,
}
```

## Minimal Code Template To Adapt (Policy)

```python
@override
def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
    # `prefix_text` is expected to be returned from the jitted model path.
    prefix_text = None

    inputs = jax.tree.map(lambda x: x, obs)
    inputs = self._input_transform(inputs)

    # ... existing batching/model call logic unchanged ...
    sample_result = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
    if isinstance(sample_result, dict):
        actions = sample_result["actions"]
        prefix_text = sample_result.get("prefix_text")
    else:
        actions = sample_result

    outputs = {"state": inputs["state"], "actions": actions}
    outputs = self._output_transform(outputs)

    if prefix_text is not None:
        outputs["prefix_text"] = prefix_text

    outputs["policy_timing"] = {"infer_ms": model_time * 1000}
    return outputs
```

## Important

- Keep the answer tightly tied to OpenPI paths and APIs.
- Do not provide generic transformer-only advice without mapping it to this repo.

## Reminder: Potential Problem

- Early string insertion in policy can break numeric mapping logic.
- If `prefix_text` is added before tensor-to-numpy conversion or before `output_transforms`, array-only operations can fail.
- Keep model and transform outputs numeric first, then attach `prefix_text` at the end of policy output assembly.
