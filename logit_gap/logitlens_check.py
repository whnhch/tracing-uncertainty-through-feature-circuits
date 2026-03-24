import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from logit_gap import (
    compute_feature_logit_attributions,
    compute_logit_gaps,
)

# -------- CONFIG --------
MODEL_NAME = "gpt2"
DECODER_PATH = "decoder_directions.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = -1

PROMPTS = [
    ("1 + 1 =", " 2"),
    ("The capital of France is", " Paris"),
    ("The meaning of life is", " 42"),
]
# ------------------------


def load_decoder(path):
    obj = torch.load(path, map_location=DEVICE)
    if isinstance(obj, dict):
        for k in ["decoder_directions", "W_dec", "decoder"]:
            if k in obj:
                return obj[k].to(DEVICE)
    return obj.to(DEVICE)


def get_hidden_and_logits(model, input_ids):
    outputs = model(input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[LAYER]  # [1, seq, d_model]
    logits = outputs.logits  # [1, seq, vocab]
    return hidden[:, -1, :], logits[:, -1, :]


def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)


def run_example(model, tokenizer, decoder, unembedding, prompt, target):
    tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = tokens["input_ids"]

    target_id = tokenizer.encode(target, add_special_tokens=False)
    assert len(target_id) == 1, f"Target must be 1 token: {target}"
    target_id = target_id[0]

    # ---- model forward ----
    h, logits = get_hidden_and_logits(model, input_ids)

    # ---- LogitLens entropy ----
    entropy = compute_entropy(logits).item()

    # ---- feature activations (dummy for now) ----
    z = h @ decoder.T
    z = z.unsqueeze(1)

    # ---- feature logit gap ----
    feature_attr = compute_feature_logit_attributions(z, decoder, unembedding)

    result = compute_logit_gaps(feature_attr, target_id)
    gaps = result.logit_gaps[0, 0]

    # normalize
    gaps_norm = gaps / (gaps.abs().max() + 1e-8)

    return entropy, gaps_norm.detach().cpu().numpy()


def main():
    print("\n=== Loading model ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    decoder = load_decoder(DECODER_PATH)
    unembedding = model.get_output_embeddings().weight.T.to(DEVICE)

    n = len(PROMPTS)

    plt.figure(figsize=(5 * n, 4))

    for i, (prompt, target) in enumerate(PROMPTS):
        entropy, gaps = run_example(
            model, tokenizer, decoder, unembedding, prompt, target
        )

        print("\n---")
        print(f"Prompt: {prompt}")
        print(f"Target: {target}")
        print(f"LogitLens Entropy: {entropy:.4f}")

        plt.subplot(1, n, i + 1)
        plt.hist(gaps, bins=50)
        plt.title(f"Entropy: {entropy:.2f}\n{prompt}")
        plt.xlabel("Normalized Logit Gap")
        plt.ylabel("# Features")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
