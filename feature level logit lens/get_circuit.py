import argparse
import json
import re
import time
from collections import deque
from collections import defaultdict
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def layer_num_from_key(layer_key: str) -> int:
	try:
		return int(layer_key.split("-", 1)[0])
	except (TypeError, ValueError, AttributeError):
		raise ValueError(f"Invalid layer key format: {layer_key}")


def feature_node_id(layer_num: int, feature_id: int) -> str:
	return f"L{layer_num}_F{feature_id}"


def normalize_words(text: str) -> set[str]:
	words = re.findall(r"[a-z0-9]+", str(text).lower())
	return set(words)


def jaccard(a: set[str], b: set[str]) -> float:
	if not a or not b:
		return 0.0
	u = a | b
	if not u:
		return 0.0
	return len(a & b) / len(u)


def parse_target_ref(target_ref: Any, default_layer_num: int) -> tuple[int, int] | None:
	# Neuronpedia usually returns integer indices (same source/layer).
	if isinstance(target_ref, int):
		return default_layer_num, target_ref

	# Defensive support for richer target references in case API shape changes.
	if isinstance(target_ref, str):
		if ":" in target_ref:
			left, right = target_ref.split(":", 1)
			if left.isdigit() and right.isdigit():
				return int(left), int(right)
		if target_ref.isdigit():
			return default_layer_num, int(target_ref)

	if isinstance(target_ref, dict):
		layer = target_ref.get("layer") or target_ref.get("layerNum")
		index = target_ref.get("index") or target_ref.get("feature") or target_ref.get("featureId")
		if isinstance(layer, int) and isinstance(index, int):
			return layer, index

	return None


def fetch_feature(model_id: str, layer_key: str, feature_id: int) -> dict | None:
	url = f"https://www.neuronpedia.org/api/feature/{model_id}/{layer_key}/{feature_id}"
	try:
		with urlopen(url, timeout=30) as resp:
			return json.loads(resp.read().decode("utf-8"))
	except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
		return None


def extract_explanations(record: dict, max_items: int = 3) -> list[str]:
	items = record.get("explanations") or []
	texts: list[str] = []

	for item in items:
		if isinstance(item, str):
			text = item.strip()
		elif isinstance(item, dict):
			text = str(item.get("description") or item.get("text") or item.get("explanation") or "").strip()
		else:
			text = ""

		if text:
			texts.append(text)
		if len(texts) >= max_items:
			break

	return texts


def paired_token_scores(tokens: list[Any], values: list[Any], top_k: int) -> list[dict]:
	pairs: list[dict] = []
	for idx in range(min(len(tokens), len(values), top_k)):
		value = None
		try:
			value = float(values[idx])
		except (TypeError, ValueError):
			value = None

		pairs.append(
			{
				"token": str(tokens[idx]),
				"score": value,
			}
		)

	return pairs


def extract_feature_logit_view(record: dict, top_k: int = 10) -> dict:
	return {
		"top_positive": paired_token_scores(
			tokens=record.get("pos_str") or [],
			values=record.get("pos_values") or [],
			top_k=top_k,
		),
		"top_negative": paired_token_scores(
			tokens=record.get("neg_str") or [],
			values=record.get("neg_values") or [],
			top_k=top_k,
		),
	}


def choose_seed_feature_per_layer(
	payload: dict,
	example_text: str,
	feature_ids_by_layer: dict[str, list[int]],
	prompts_by_layer: dict[str, list[str]],
) -> list[dict]:
	example_tokens = normalize_words(example_text)
	seeds: list[dict] = []

	for layer_key, feature_ids in feature_ids_by_layer.items():
		layer_num = layer_num_from_key(layer_key)
		prompts = prompts_by_layer.get(layer_key, [])
		best_idx = None
		best_score = -1.0

		for idx in range(min(len(feature_ids), len(prompts))):
			score = jaccard(example_tokens, normalize_words(prompts[idx]))
			if score > best_score:
				best_score = score
				best_idx = idx

		if best_idx is None:
			continue

		seeds.append(
			{
				"id": feature_node_id(layer_num, feature_ids[best_idx]),
				"layer": layer_num,
				"layer_key": layer_key,
				"feature_id": feature_ids[best_idx],
				"prompt": prompts[best_idx],
				"similarity_to_example": round(best_score, 6),
			}
		)

	seeds.sort(key=lambda x: x["layer"])
	return seeds


def append_weighted_edges(
	edges: list[dict],
	src_layer_num: int,
	src_feature_id: int,
	target_refs: list[Any],
	weights: list[float],
	edge_type: str,
) -> None:
	source = feature_node_id(src_layer_num, src_feature_id)

	for idx, target_ref in enumerate(target_refs):
		parsed = parse_target_ref(target_ref, default_layer_num=src_layer_num)
		if parsed is None:
			continue

		tgt_layer_num, tgt_feature_id = parsed
		weight = None
		if idx < len(weights):
			try:
				weight = float(weights[idx])
			except (TypeError, ValueError):
				weight = None

		edges.append(
			{
				"source": source,
				"target": feature_node_id(tgt_layer_num, tgt_feature_id),
				"source_layer": src_layer_num,
				"target_layer": tgt_layer_num,
				"source_feature": src_feature_id,
				"target_feature": tgt_feature_id,
				"edge_type": edge_type,
				"weight": weight,
			}
		)


def summarize(nodes: list[dict], edges: list[dict], selected_layers: list[int], failures: list[dict]) -> dict:
	nodes_per_layer: dict[int, int] = defaultdict(int)
	for node in nodes:
		nodes_per_layer[node["layer"]] += 1

	edges_per_type: dict[str, int] = defaultdict(int)
	same_layer_edges = 0
	cross_layer_edges = 0
	for edge in edges:
		edges_per_type[edge["edge_type"]] += 1
		if edge["source_layer"] == edge["target_layer"]:
			same_layer_edges += 1
		else:
			cross_layer_edges += 1

	return {
		"layer_order": selected_layers,
		"node_count": len(nodes),
		"edge_count": len(edges),
		"same_layer_edge_count": same_layer_edges,
		"cross_layer_edge_count": cross_layer_edges,
		"nodes_per_layer": dict(sorted(nodes_per_layer.items())),
		"edges_per_type": dict(edges_per_type),
		"failed_fetch_count": len(failures),
	}


def record_to_node(
	record: dict,
	layer_key: str,
	layer_num: int,
	feature_id: int,
	prompt_lookup: dict[tuple[str, int], str],
	logit_top_k: int,
	max_explanations: int,
) -> dict:
	explanations = extract_explanations(record, max_items=max(0, max_explanations))
	logit_view = extract_feature_logit_view(record, top_k=max(0, logit_top_k))

	return {
		"id": feature_node_id(layer_num, feature_id),
		"layer": layer_num,
		"layer_key": layer_key,
		"feature_id": feature_id,
		"prompt": prompt_lookup.get((layer_key, feature_id)),
		"max_act_approx": record.get("maxActApprox"),
		"hook_name": record.get("hookName"),
		"feature_explanations": explanations,
		"feature_logits": logit_view,
	}


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Extract Neuronpedia circuit graph fields for sampled features with rate limiting."
	)
	parser.add_argument("--input", default="random_prompts.json", help="Path to input JSON")
	parser.add_argument("--output", default="circuit.json", help="Path to output JSON")
	parser.add_argument(
		"--sleep-seconds",
		type=float,
		default=0.8,
		help="Delay between API calls to avoid sending bursts",
	)
	parser.add_argument(
		"--max-features-per-layer",
		type=int,
		default=0,
		help="Limit features per layer (0 means all sampled features)",
	)
	parser.add_argument(
		"--example-index",
		type=int,
		default=-1,
		help="Prompt index in payload['prompts'] for per-example circuit mode (>=0 enables it)",
	)
	parser.add_argument(
		"--hop-depth",
		type=int,
		default=1,
		help="Neighbor expansion depth from each seed feature in per-example mode",
	)
	parser.add_argument(
		"--topk-neighbors",
		type=int,
		default=25,
		help="Max topkCosSim neighbors to traverse from each fetched feature",
	)
	parser.add_argument(
		"--logit-top-k",
		type=int,
		default=10,
		help="Number of positive/negative logit tokens to keep per feature",
	)
	parser.add_argument(
		"--max-explanations",
		type=int,
		default=3,
		help="Number of explanation strings to keep per feature",
	)
	args = parser.parse_args()

	with open(args.input, "r", encoding="utf-8") as f:
		payload = json.load(f)

	model_id = payload.get("model")
	selected_layers = sorted(payload.get("selected_layers", []))
	feature_ids_by_layer: dict[str, list[int]] = payload.get("sampled_feature_ids_by_layer", {})
	prompts_by_layer: dict[str, list[str]] = payload.get("prompts_by_layer", {})
	all_prompts: list[str] = payload.get("prompts", [])

	prompt_lookup: dict[tuple[str, int], str] = {}
	for layer_key, feature_ids in feature_ids_by_layer.items():
		prompts = prompts_by_layer.get(layer_key, [])
		for i in range(min(len(feature_ids), len(prompts))):
			prompt_lookup[(layer_key, feature_ids[i])] = prompts[i]

	nodes: list[dict] = []
	edges: list[dict] = []
	failures: list[dict] = []

	seed_features: list[dict] = []

	if args.example_index >= 0:
		if args.example_index >= len(all_prompts):
			raise ValueError(f"example-index {args.example_index} is out of range for prompts size {len(all_prompts)}")

		example_text = all_prompts[args.example_index]
		seed_features = choose_seed_feature_per_layer(
			payload=payload,
			example_text=example_text,
			feature_ids_by_layer=feature_ids_by_layer,
			prompts_by_layer=prompts_by_layer,
		)

		visited: set[tuple[str, int]] = set()
		queue: deque[tuple[str, int, int]] = deque()
		nodes_by_id: dict[str, dict] = {}

		for seed in seed_features:
			queue.append((seed["layer_key"], seed["feature_id"], 0))

		while queue:
			layer_key, feature_id, depth = queue.popleft()
			key = (layer_key, feature_id)
			if key in visited:
				continue
			visited.add(key)

			record = fetch_feature(model_id=model_id, layer_key=layer_key, feature_id=feature_id)
			if record is None:
				failures.append({"layer": layer_key, "feature": feature_id})
				time.sleep(args.sleep_seconds)
				continue

			layer_num = layer_num_from_key(layer_key)
			node = record_to_node(
				record=record,
				layer_key=layer_key,
				layer_num=layer_num,
				feature_id=feature_id,
				prompt_lookup=prompt_lookup,
				logit_top_k=args.logit_top_k,
				max_explanations=args.max_explanations,
			)
			nodes_by_id[node["id"]] = node

			topk_indices = (record.get("topkCosSimIndices") or [])[: max(0, args.topk_neighbors)]
			topk_values = (record.get("topkCosSimValues") or [])[: max(0, args.topk_neighbors)]

			append_weighted_edges(
				edges=edges,
				src_layer_num=layer_num,
				src_feature_id=feature_id,
				target_refs=topk_indices,
				weights=topk_values,
				edge_type="topk_cosine_feature",
			)
			append_weighted_edges(
				edges=edges,
				src_layer_num=layer_num,
				src_feature_id=feature_id,
				target_refs=record.get("correlated_features_indices") or [],
				weights=record.get("correlated_features_pearson") or [],
				edge_type="correlated_feature_pearson",
			)
			append_weighted_edges(
				edges=edges,
				src_layer_num=layer_num,
				src_feature_id=feature_id,
				target_refs=record.get("correlated_features_indices") or [],
				weights=record.get("correlated_features_l1") or [],
				edge_type="correlated_feature_l1",
			)

			if depth < max(0, args.hop_depth):
				for target_ref in topk_indices:
					parsed = parse_target_ref(target_ref, default_layer_num=layer_num)
					if parsed is None:
						continue
					tgt_layer_num, tgt_feature_id = parsed
					tgt_layer_key = f"{tgt_layer_num}-gemmascope-res-16k"
					queue.append((tgt_layer_key, tgt_feature_id, depth + 1))

			time.sleep(args.sleep_seconds)

		nodes = sorted(nodes_by_id.values(), key=lambda n: (n["layer"], n["feature_id"]))
	else:
		for layer_key, feature_ids in feature_ids_by_layer.items():
			layer_num = layer_num_from_key(layer_key)
			if args.max_features_per_layer and args.max_features_per_layer > 0:
				feature_ids = feature_ids[: args.max_features_per_layer]

			for feature_id in feature_ids:
				record = fetch_feature(model_id=model_id, layer_key=layer_key, feature_id=feature_id)
				if record is None:
					failures.append({"layer": layer_key, "feature": feature_id})
					time.sleep(args.sleep_seconds)
					continue

				nodes.append(
					record_to_node(
						record=record,
						layer_key=layer_key,
						layer_num=layer_num,
						feature_id=feature_id,
						prompt_lookup=prompt_lookup,
						logit_top_k=args.logit_top_k,
						max_explanations=args.max_explanations,
					)
				)

				append_weighted_edges(
					edges=edges,
					src_layer_num=layer_num,
					src_feature_id=feature_id,
					target_refs=(record.get("topkCosSimIndices") or [])[: max(0, args.topk_neighbors)],
					weights=(record.get("topkCosSimValues") or [])[: max(0, args.topk_neighbors)],
					edge_type="topk_cosine_feature",
				)
				append_weighted_edges(
					edges=edges,
					src_layer_num=layer_num,
					src_feature_id=feature_id,
					target_refs=record.get("correlated_features_indices") or [],
					weights=record.get("correlated_features_pearson") or [],
					edge_type="correlated_feature_pearson",
				)
				append_weighted_edges(
					edges=edges,
					src_layer_num=layer_num,
					src_feature_id=feature_id,
					target_refs=record.get("correlated_features_indices") or [],
					weights=record.get("correlated_features_l1") or [],
					edge_type="correlated_feature_l1",
				)

				time.sleep(args.sleep_seconds)

	summary = summarize(nodes=nodes, edges=edges, selected_layers=selected_layers, failures=failures)
	result = {
		"model": model_id,
		"method": "neuronpedia_feature_graph_fields",
		"mode": "per_example" if args.example_index >= 0 else "all_sampled_features",
		"input_file": args.input,
		"parameters": {
			"sleep_seconds": args.sleep_seconds,
			"max_features_per_layer": args.max_features_per_layer,
			"example_index": args.example_index,
			"hop_depth": args.hop_depth,
			"topk_neighbors": args.topk_neighbors,
			"logit_top_k": args.logit_top_k,
			"max_explanations": args.max_explanations,
		},
		"example_prompt": all_prompts[args.example_index] if args.example_index >= 0 else None,
		"seed_features": seed_features,
		"summary": summary,
		"nodes": nodes,
		"edges": edges,
		"fetch_failures": failures,
	}

	with open(args.output, "w", encoding="utf-8") as f:
		json.dump(result, f, indent=2, ensure_ascii=False)

	print(f"Wrote {args.output}")
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()
