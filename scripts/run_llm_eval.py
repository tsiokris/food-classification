import argparse
import base64
import io
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from src.data.config import ROTATION_0, ROTATION_1, ROTATION_2
from src.data.dataset import load_image

load_dotenv()

ROTATIONS = [ROTATION_0, ROTATION_1, ROTATION_2]

SYSTEM_PROMPT = (
    "You are a food classification expert. "
    "You will be shown a food image and a list of possible food classes. "
    "Respond ONLY with a JSON object in this exact format:\n"
    '{"predictions": ["class1", "class2", "class3", "class4", "class5"]}\n'
    "List your top 5 predictions ranked from most to least likely. "
    "Use exact class names from the provided list. No explanation, no extra text."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True,              help="Model name (e.g. deepseek-vl2)")
    p.add_argument("--base-url",    required=True,              help="API base URL")
    p.add_argument("--api-key-env", default="LLM_API_KEY",      help="Env var holding the API key")
    p.add_argument("--folds",       nargs="+", type=int, default=[0, 1, 2], choices=[0, 1, 2])
    p.add_argument("--delay",       type=float, default=0.5,    help="Seconds to wait between API calls")
    p.add_argument("--output",      default=None,               help="Output JSON path (default: results/llm_{model}.json)")
    return p.parse_args()


def encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def build_user_prompt(labels: list) -> str:
    label_list = "\n".join(f"- {l}" for l in sorted(labels))
    return (
        f"Identify the food in this image. Choose from these classes:\n\n{label_list}\n\n"
        "Return your top 5 predictions as a JSON object with key 'predictions'."
    )


def call_api(client: OpenAI, model: str, img: Image.Image, user_prompt: str) -> list:
    b64 = encode_image(img)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        temperature=0,
    )
    text = response.choices[0].message.content.strip()
    parsed = json.loads(text)
    return parsed["predictions"][:5]


def resolve_path(row, config) -> str:
    path = row[config.image_col]
    if config.source_col:
        root = config.source_roots.get(row[config.source_col], "")
        return os.path.join(root, path) if root else path
    return path


def compute_metrics(records: list) -> dict:
    valid = [r for r in records if r["predictions"] is not None]
    n = len(valid)
    if n == 0:
        return {"top1": 0.0, "top3": 0.0, "top5": 0.0, "n_samples": 0, "n_errors": len(records)}
    return {
        "top1":     round(sum(1 for r in valid if r["gt"] in r["predictions"][:1]) / n, 4),
        "top3":     round(sum(1 for r in valid if r["gt"] in r["predictions"][:3]) / n, 4),
        "top5":     round(sum(1 for r in valid if r["gt"] in r["predictions"][:5]) / n, 4),
        "n_samples": n,
        "n_errors":  len(records) - n,
    }


def evaluate_fold(fold_idx: int, config, client: OpenAI, model: str, user_prompt: str, delay: float, cache_path: str) -> dict:
    # load cache so we can resume if interrupted
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                r = json.loads(line)
                cache[r["image_path"]] = r
        print(f"  Resuming: {len(cache)} images already cached")

    df = pd.read_csv(config.csv_path)
    test_df = df[df[config.split_col].isin(config.test_splits)].reset_index(drop=True)
    n = len(test_df)
    print(f"  Test samples: {n}")

    with open(cache_path, "a") as cache_file:
        for _, row in test_df.iterrows():
            img_path = resolve_path(row, config)
            if img_path in cache:
                continue

            gt = row[config.label_col]
            try:
                img = load_image(img_path)
                predictions = call_api(client, model, img, user_prompt)
                record = {"image_path": img_path, "gt": gt, "predictions": predictions}
            except Exception as e:
                print(f"  [error] {img_path}: {e}")
                record = {"image_path": img_path, "gt": gt, "predictions": None, "error": str(e)}

            cache[img_path] = record
            cache_file.write(json.dumps(record) + "\n")
            cache_file.flush()

            done = len(cache)
            if done % 50 == 0 or done == n:
                valid = [r for r in cache.values() if r["predictions"] is not None]
                running_top1 = sum(1 for r in valid if r["gt"] in r["predictions"][:1]) / max(len(valid), 1)
                print(f"  [{done}/{n}]  running top1: {running_top1:.3f}")

            if delay > 0:
                time.sleep(delay)

    return compute_metrics(list(cache.values()))


def main():
    args = parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"Error: environment variable '{args.api_key_env}' is not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    labels = pd.read_csv("cnn_splits/classes.csv")["name"].tolist()
    user_prompt = build_user_prompt(labels)

    model_slug = args.model.replace("/", "_").replace(":", "_")
    output_path = args.output or f"results/llm_{model_slug}.json"
    os.makedirs("results", exist_ok=True)

    print(f"Model:   {args.model}")
    print(f"API:     {args.base_url}")
    print(f"Folds:   {args.folds}")
    print(f"Output:  {output_path}\n")

    fold_results = {}

    for fold_idx in args.folds:
        config = ROTATIONS[fold_idx]
        print(f"{'=' * 60}")
        print(f"Fold {fold_idx}  ({config.name})")
        print(f"{'=' * 60}")

        cache_path = f"results/llm_{model_slug}_fold{fold_idx}_cache.jsonl"
        metrics = evaluate_fold(fold_idx, config, client, args.model, user_prompt, args.delay, cache_path)
        fold_results[str(fold_idx)] = {"config": config.name, **metrics}

        print(f"  top1: {metrics['top1']*100:.2f}%  top3: {metrics['top3']*100:.2f}%  top5: {metrics['top5']*100:.2f}%\n")

    avg = {
        k: round(sum(fold_results[str(i)][k] for i in args.folds) / len(args.folds), 4)
        for k in ["top1", "top3", "top5"]
    }

    result = {
        "model":     args.model,
        "base_url":  args.base_url,
        "timestamp": datetime.utcnow().isoformat(),
        "folds":     fold_results,
        "avg":       avg,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {output_path}")
    print(f"\nAverage across folds:")
    for k in ["top1", "top3", "top5"]:
        print(f"  {k}: {avg[k]*100:.2f}%")


if __name__ == "__main__":
    main()
