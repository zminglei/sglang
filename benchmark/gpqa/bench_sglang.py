"""
GPQA Diamond benchmark for SGLang.
Usage:
    python benchmark/gpqa/bench_sglang.py --data-path /shared/user/zminglei/gpqa_diamond.jsonl

Compare FP8 KV cache vs BF16:
    # Server 1: python3 -m sglang.launch_server --model-path <model> --attention-backend fa3 --kv-cache-dtype fp8_e4m3
    # Server 2: python3 -m sglang.launch_server --model-path <model> --attention-backend fa3
    # Then run this script against each server.
"""

import argparse
import json
import random
import re
import time

import numpy as np

from sglang.utils import read_jsonl


CHOICE_LABELS = ["A", "B", "C", "D"]


def load_gpqa_data(data_path):
    lines = list(read_jsonl(data_path))
    examples = []
    for line in lines:
        question = line["Question"]
        correct = line["Correct Answer"]
        incorrect = [
            line["Incorrect Answer 1"],
            line["Incorrect Answer 2"],
            line["Incorrect Answer 3"],
        ]
        # Shuffle choices with a fixed seed per question for reproducibility
        choices = [correct] + incorrect
        rng = random.Random(hash(question) & 0xFFFFFFFF)
        rng.shuffle(choices)
        correct_idx = choices.index(correct)
        examples.append({
            "question": question,
            "choices": choices,
            "correct_label": CHOICE_LABELS[correct_idx],
        })
    return examples


def format_prompt(example):
    prompt = (
        "Answer the following multiple choice question. "
        "Think step by step and then output your answer as "
        '"The answer is (X)" where X is A, B, C, or D.\n\n'
    )
    prompt += f"Question: {example['question']}\n\n"
    for i, choice in enumerate(example["choices"]):
        prompt += f"({CHOICE_LABELS[i]}) {choice}\n"
    prompt += "\nAnswer:"
    return prompt


def extract_answer(text):
    """Extract the answer letter from model output."""
    # Look for "The answer is (X)" pattern
    match = re.search(r"[Tt]he answer is \(?([A-D])\)?", text)
    if match:
        return match.group(1)
    # Fallback: last occurrence of a standalone A/B/C/D
    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[-1]
    return None


def main(args):
    from sglang.lang.api import set_default_backend
    from sglang.test.test_utils import (
        add_common_sglang_args_and_parse,
        select_sglang_backend,
    )

    set_default_backend(select_sglang_backend(args))

    # Load data
    examples = load_gpqa_data(args.data_path)
    if args.num_questions:
        examples = examples[: args.num_questions]

    print(f"Running GPQA Diamond benchmark with {len(examples)} questions")

    # Build prompts
    prompts = [format_prompt(ex) for ex in examples]
    labels = [ex["correct_label"] for ex in examples]
    arguments = [{"question": p} for p in prompts]

    import sglang as sgl

    @sgl.function
    def gpqa_solve(s, question):
        s += question
        s += sgl.gen(
            "answer",
            max_tokens=args.max_new_tokens,
            stop=["Question:", "\n\nQ:"],
        )

    # Run
    tic = time.perf_counter()
    states = gpqa_solve.run_batch(
        arguments,
        temperature=args.temperature,
        top_p=args.top_p,
        num_threads=args.parallel,
        progress_bar=True,
    )
    latency = time.perf_counter() - tic

    # Evaluate
    preds = []
    correct = 0
    invalid = 0
    for i, state in enumerate(states):
        answer_text = state["answer"]
        pred = extract_answer(answer_text)
        preds.append(pred)
        if pred is None:
            invalid += 1
        elif pred == labels[i]:
            correct += 1

    total = len(examples)
    acc = correct / total
    inv_rate = invalid / total

    # Compute throughput
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"] for s in states
    )
    output_throughput = num_output_tokens / latency

    print(f"Accuracy: {acc:.3f} ({correct}/{total})")
    print(f"Invalid: {inv_rate:.3f} ({invalid}/{total})")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")

    # Save detailed results
    if args.result_file:
        with open(args.result_file, "a") as f:
            result = {
                "task": "gpqa_diamond",
                "backend": args.backend,
                "accuracy": round(acc, 3),
                "invalid": round(inv_rate, 3),
                "latency": round(latency, 3),
                "num_questions": total,
                "output_throughput": round(output_throughput, 3),
            }
            f.write(json.dumps(result) + "\n")

    # Save raw outputs for diffing between runs
    if args.raw_output_file:
        with open(args.raw_output_file, "w") as f:
            for i, state in enumerate(states):
                f.write(json.dumps({
                    "idx": i,
                    "pred": preds[i],
                    "label": labels[i],
                    "correct": preds[i] == labels[i],
                    "output": state["answer"],
                }) + "\n")
        print(f"Raw outputs saved to {args.raw_output_file}")


if __name__ == "__main__":
    from sglang.test.test_utils import add_common_sglang_args_and_parse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--num-questions", type=int, default=None,
                        help="Number of questions (default: all 198)")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--raw-output-file", type=str, default=None,
                        help="Save raw outputs for diffing between configs")
    args = add_common_sglang_args_and_parse(parser)
    main(args)
