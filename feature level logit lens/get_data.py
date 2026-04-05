import json
import random
import re
import requests


def detokenize(tokens: list[str]) -> str:
    # Neuronpedia tokens often use ▁ for whitespace and include BOS/EOS markers.
    return "".join(tok.replace("▁", " ") for tok in tokens if tok not in {"<bos>", "<eos>"}).strip()


def activation_score(act: dict) -> float:
    value = act.get("maxValue")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_english_like(text: str, min_letter_ratio: float = 0.7) -> bool:
    # Reject clearly non-English scripts first (Hangul/CJK).
    if re.search(r"[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF\u4E00-\u9FFF]", text):
        return False

    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False

    english_letters = [ch for ch in letters if ("a" <= ch.lower() <= "z")]
    ratio = len(english_letters) / len(letters)
    return ratio >= min_letter_ratio


def classify_prompt_style(text: str) -> str | None:
    lowered = text.lower()

    fact_patterns = [
        r"\bfact\s*:",
        r"\bq\s*:\s*.*\ba\s*:",
        r"\b(question|answer)\s*:\s*",
        r"\b(capital of|largest|smallest|population of|located in|is the|was born in|who is|what is)\b",
    ]
    if any(re.search(p, lowered, flags=re.DOTALL) for p in fact_patterns):
        return "fact"

    equation_patterns = [
        r"\bwhat\s+is\s+[-+]?\d+(?:\.\d+)?\s*[+\-*/]\s*[-+]?\d+(?:\.\d+)?\b",
        r"\b[-+]?\d+(?:\.\d+)?\s*[+\-*/]\s*[-+]?\d+(?:\.\d+)?\s*=\s*[-+]?\d+(?:\.\d+)?\b",
        r"^\s*(solve|simplify|calculate|evaluate)\b",
        r"\b(\d+\s*[+\-*/]\s*\d+|\d+\s*=\s*\d+)\b",
    ]
    if any(re.search(p, lowered) for p in equation_patterns):
        return "equation"

    return None


def is_valid_layer(model_id: str, layer_num: int) -> bool:
    layer_id = f"{layer_num}-gemmascope-res-16k"
    url = f"https://www.neuronpedia.org/api/feature/{model_id}/{layer_id}/0"
    try:
        res = requests.get(url, timeout=20)
        return res.status_code == 200
    except requests.RequestException:
        return False


def collect_unique_prompts_from_layer(
    model_id: str,
    layer_num: int,
    feature_count: int,
    target_count: int,
    min_chars: int,
    max_chars: int,
    allowed_styles: set[str],
    require_english: bool,
    global_seen: set[str],
) -> tuple[list[str], list[int]]:
    layer_id = f"{layer_num}-gemmascope-res-16k"
    feature_ids = list(range(feature_count))
    random.shuffle(feature_ids)

    prompts: list[str] = []
    used_feature_ids: list[int] = []

    for f_id in feature_ids:
        if len(prompts) >= target_count:
            break

        url = f"https://www.neuronpedia.org/api/feature/{model_id}/{layer_id}/{f_id}"
        try:
            res = requests.get(url, timeout=20)
            res.raise_for_status()
        except requests.RequestException:
            continue

        data = res.json()
        activations = data.get("activations") or []
        if not activations:
            continue

        sorted_activations = sorted(activations, key=activation_score, reverse=True)
        for act in sorted_activations:
            tokens = act.get("tokens") or []
            text = normalize_text(detokenize(tokens))
            if not text:
                continue
            if len(text) < min_chars or len(text) > max_chars:
                continue
            if require_english and not is_english_like(text):
                continue

            style = classify_prompt_style(text)
            if style is None or style not in allowed_styles:
                continue
            if text in global_seen:
                continue

            prompts.append(text)
            used_feature_ids.append(f_id)
            global_seen.add(text)
            break

    return prompts, used_feature_ids


MIN_PROMPT_CHARS = 8
MAX_PROMPT_CHARS = 200
FEATURE_COUNT = 16384
MODEL_ID = "gemma-2-2b"
REQUIRE_ENGLISH = True
ALLOWED_STYLES = {"fact", "equation"}
LAYER_COUNT = 5
REQUIRED_LAYER = 21
SAMPLES_PER_LAYER = 20
LAYER_CANDIDATES = list(range(42))

# Discover valid layers for this model and enforce including L21.
valid_layers = [layer for layer in LAYER_CANDIDATES if is_valid_layer(MODEL_ID, layer)]
if REQUIRED_LAYER not in valid_layers:
    raise RuntimeError(f"Required layer {REQUIRED_LAYER} is not available for model {MODEL_ID}.")

other_layers = [layer for layer in valid_layers if layer != REQUIRED_LAYER]
if len(other_layers) < LAYER_COUNT - 1:
    raise RuntimeError("Not enough valid layers to sample the requested count.")

selected_layers = [REQUIRED_LAYER] + random.sample(other_layers, LAYER_COUNT - 1)
selected_layers.sort()

global_seen_texts: set[str] = set()
prompts_by_layer: dict[str, list[str]] = {}
feature_ids_by_layer: dict[str, list[int]] = {}

for layer_num in selected_layers:
    prompts, used_features = collect_unique_prompts_from_layer(
        model_id=MODEL_ID,
        layer_num=layer_num,
        feature_count=FEATURE_COUNT,
        target_count=SAMPLES_PER_LAYER,
        min_chars=MIN_PROMPT_CHARS,
        max_chars=MAX_PROMPT_CHARS,
        allowed_styles=ALLOWED_STYLES,
        require_english=REQUIRE_ENGLISH,
        global_seen=global_seen_texts,
    )
    layer_key = f"{layer_num}-gemmascope-res-16k"
    prompts_by_layer[layer_key] = prompts
    feature_ids_by_layer[layer_key] = used_features
    fact_count = sum(1 for prompt in prompts if classify_prompt_style(prompt) == "fact")
    equation_count = sum(1 for prompt in prompts if classify_prompt_style(prompt) == "equation")
    print(f"{layer_key}: {len(prompts)} prompts (fact={fact_count}, equation={equation_count})")

all_random_inputs = [prompt for layer_key in prompts_by_layer for prompt in prompts_by_layer[layer_key]]
print(f"Total prompt #: {len(all_random_inputs)}")

output_path = "random_prompts.json"
payload = {
    "model": MODEL_ID,
    "selected_layers": selected_layers,
    "required_layer": REQUIRED_LAYER,
    "samples_per_layer_target": SAMPLES_PER_LAYER,
    "layer_count_target": LAYER_COUNT,
    "sampled_feature_ids_by_layer": feature_ids_by_layer,
    "prompts_by_layer": prompts_by_layer,
    "prompt_count": len(all_random_inputs),
    "prompts": all_random_inputs,
    "min_prompt_chars": MIN_PROMPT_CHARS,
    "max_prompt_chars": MAX_PROMPT_CHARS,
    "require_english": REQUIRE_ENGLISH,
    "allowed_styles": sorted(ALLOWED_STYLES),
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)