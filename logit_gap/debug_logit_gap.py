import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from logit_gap import (
    compute_feature_logit_attributions,
    compute_logit_gaps,
)

# -------- CONFIG --------
MODEL_NAME = "gpt2"
DECODER_PATH = "decoder_directions.pt"

PROMPT = "1 + 1 ="
TARGET = " 2"

TOP_K = 10
LAYER = -1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------


def load_decoder(path):
    obj = torch.load(path, map_location=DEVICE)
    if isinstance(obj, dict):
        for k in ["decoder_directions", "W_dec", "decoder"]:
            if k in obj:
                return obj[k].to(DEVICE)
    return obj.to(DEVICE)


def get_hidden(model, input_ids):
    out = model(input_ids, output_hidden_states=True)
    h = out.hidden_states[LAYER]  # [1, seq, d_model]
    return h[:, -1, :]  # last token


def main():
    print("\n=== Loading model ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    decoder = load_decoder(DECODER_PATH)
    unembedding = model.get_output_embeddings().weight.T.to(DEVICE)

    tokens = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
    input_ids = tokens["input_ids"]

    target_id = tokenizer.encode(TARGET, add_special_tokens=False)
    assert len(target_id) == 1, "Target must be single token"
    target_id = target_id[0]

    print(f"\nPrompt: {PROMPT}")
    print(f"Target token: {TARGET} (id={target_id})")

    # hidden state
    h = get_hidden(model, input_ids)

    # dummy feature activations
    z = h @ decoder.T
    z = z.unsqueeze(1)

    # compute
    feature_attr = compute_feature_logit_attributions(z, decoder, unembedding)

    result = compute_logit_gaps(feature_attr, target_id)

    gaps = result.logit_gaps[0, 0]
    target_logits = result.target_logits[0, 0]
    competitor_logits = result.competitor_logits[0, 0]

    # -------- CHECK --------
    print("\n=== CHECK 1: Manual vs computed gap ===")
    i = 0
    logits_i = feature_attr[0, 0, i]

    manual_target = logits_i[target_id]
    masked = torch.cat([logits_i[:target_id], logits_i[target_id + 1 :]])
    manual_comp = masked.max()

    manual_gap = manual_target - manual_comp
    computed_gap = gaps[i]

    print(f"Feature {i}")
    print(f"Manual gap:   {manual_gap.item():.6f}")
    print(f"Computed gap: {computed_gap.item():.6f}")

    # -------- TABLE --------
    print("\n=== Top positive features ===")
    vals, idx = torch.topk(gaps, TOP_K)
    for v, i in zip(vals.tolist(), idx.tolist()):
        print(f"feat={i:4d} | gap={v:10.4f}")

    print("\n=== Top negative features ===")
    vals, idx = torch.topk(gaps, TOP_K, largest=False)
    for v, i in zip(vals.tolist(), idx.tolist()):
        print(f"feat={i:4d} | gap={v:10.4f}")

    # -------- NORMALIZATION (optional but helpful) --------
    gaps_norm = gaps / (gaps.abs().max() + 1e-8)

    # -------- HISTOGRAM --------
    print("\n=== Plotting gap distribution ===")
    plt.figure()
    plt.hist(gaps_norm.detach().cpu().numpy(), bins=50)

    plt.title("Normalized Logit Gap Distribution")
    plt.xlabel("Normalized Logit Gap")
    plt.ylabel("Number of Features")
    plt.show()

    # -------- SCATTER --------
    print("\n=== Plotting target vs competitor logits ===")
    plt.figure()
    plt.scatter(
        target_logits.detach().cpu().numpy(),
        competitor_logits.detach().cpu().numpy(),
        alpha=0.5,
    )
    plt.xlabel("Target Logit")
    plt.ylabel("Best Competitor Logit")
    plt.title("Feature Competition")
    plt.show()


if __name__ == "__main__":
    main()
