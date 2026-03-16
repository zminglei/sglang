"""
Test FP8 KV cache accuracy degradation at long context lengths.

Strategy:
1. "Needle in a haystack" — embed a fact early in a long context,
   pad with filler text, then ask about the fact at the end.
2. Repeat at increasing context lengths to find where FP8 breaks.
3. Also dump Q value stats if running with a debug hook.

Usage:
    # Against FP8 KV cache server:
    python benchmark/fp8_kv_long_context_test.py --port 30000 --max-ctx 16384

    # Against BF16 server:
    python benchmark/fp8_kv_long_context_test.py --port 30001 --max-ctx 16384
"""

import argparse
import json
import time

import requests


def build_needle_haystack_prompt(target_ctx_len: int, tokenizer_ratio: float = 3.5):
    """Build a needle-in-a-haystack prompt targeting a specific token count.

    Args:
        target_ctx_len: target number of tokens
        tokenizer_ratio: approximate chars per token
    """
    needle = (
        "The secret project code name is AURORA-7749. "
        "Remember this code name, it is very important."
    )

    # Filler paragraphs — boring but coherent text to pad context
    filler_paragraphs = [
        "The quarterly financial report showed steady growth across all divisions. "
        "Revenue increased by 3.2% compared to the previous quarter, driven primarily "
        "by strong performance in the enterprise segment. Operating margins remained "
        "stable at 18.4%, reflecting effective cost management strategies.",

        "The research team published their findings on advanced polymer composites "
        "in the latest issue of Materials Science Review. The study demonstrated "
        "that incorporating nano-scale reinforcements could improve tensile strength "
        "by up to 47% without significantly increasing manufacturing costs.",

        "Infrastructure development in the northern region continued according to "
        "schedule. The highway expansion project reached 68% completion, with the "
        "new interchange system expected to be operational by the end of next fiscal "
        "year. Environmental impact assessments were filed and approved.",

        "The annual employee satisfaction survey results indicated improvements in "
        "work-life balance scores, with an average rating of 4.1 out of 5.0, up "
        "from 3.7 the previous year. Remote work policies were cited as the primary "
        "driver of this improvement across all departments.",

        "Supply chain optimization efforts yielded a 12% reduction in lead times "
        "for critical components. The new vendor management system automated 85% "
        "of routine procurement tasks, freeing up the team to focus on strategic "
        "sourcing initiatives and long-term partnership development.",

        "The cybersecurity team conducted their semi-annual penetration testing "
        "exercise. Results showed significant improvement in the organization's "
        "security posture, with critical vulnerability count dropping from 23 to "
        "7 compared to the previous assessment cycle.",

        "Market analysis for the Asia-Pacific region revealed growing demand for "
        "cloud-based solutions, with projected annual growth rates of 22% through "
        "2028. The competitive landscape remained fragmented, presenting "
        "opportunities for strategic acquisitions and partnerships.",

        "Training and development programs were expanded to include AI literacy "
        "courses for all technical staff. Completion rates exceeded 90% in the "
        "first cohort, and preliminary feedback suggested high satisfaction with "
        "the curriculum content and delivery format.",
    ]

    target_chars = int(target_ctx_len * tokenizer_ratio)

    # Place needle at ~10% into the context
    needle_pos_chars = int(target_chars * 0.1)

    # Build haystack
    haystack = ""
    para_idx = 0
    while len(haystack) < needle_pos_chars:
        haystack += filler_paragraphs[para_idx % len(filler_paragraphs)] + "\n\n"
        para_idx += 1

    haystack += needle + "\n\n"

    while len(haystack) < target_chars - 200:  # leave room for question
        haystack += filler_paragraphs[para_idx % len(filler_paragraphs)] + "\n\n"
        para_idx += 1

    question = (
        "Based on everything you have read above, what is the secret project "
        "code name? Reply with ONLY the code name, nothing else."
    )

    return haystack + "\n" + question, "AURORA-7749"


def build_long_reasoning_prompt(target_ctx_len: int, tokenizer_ratio: float = 3.5):
    """Build a long math chain prompt to test reasoning coherence."""
    target_chars = int(target_ctx_len * tokenizer_ratio)

    prompt = "Let's solve a series of math problems step by step.\n\n"

    problem_num = 1
    # Add enough problems to fill context
    while len(prompt) < target_chars - 500:
        a, b = problem_num * 17 + 3, problem_num * 13 + 7
        prompt += (
            f"Problem {problem_num}: What is {a} + {b}?\n"
            f"Solution: {a} + {b} = {a + b}\n\n"
        )
        problem_num += 1

    # Final problem — model should get this right if attention works
    final_a, final_b = 8847, 3156
    prompt += (
        f"Problem {problem_num}: What is {final_a} + {final_b}?\n"
        f"Solution:"
    )
    return prompt, str(final_a + final_b)


def query_server(prompt, port, max_tokens=64):
    """Send a completion request to the sglang server."""
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["text"]
        usage = data.get("usage", {})
        return text.strip(), usage
    except Exception as e:
        return f"ERROR: {e}", {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--max-ctx", type=int, default=16384,
                        help="Maximum context length to test")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    context_lengths = [512, 1024, 2048, 4096, 8192]
    context_lengths = [c for c in context_lengths if c <= args.max_ctx]
    if args.max_ctx not in context_lengths:
        context_lengths.append(args.max_ctx)

    print(f"Testing FP8 KV cache accuracy at various context lengths")
    print(f"Server: localhost:{args.port}")
    print(f"Context lengths: {context_lengths}")
    print("=" * 80)

    results = []

    for ctx_len in context_lengths:
        print(f"\n--- Context length: ~{ctx_len} tokens ---")

        # Test 1: Needle in a haystack
        prompt, expected = build_needle_haystack_prompt(ctx_len)
        output, usage = query_server(prompt, args.port, args.max_tokens)
        prompt_tokens = usage.get("prompt_tokens", "?")
        needle_pass = expected in output
        print(f"  Needle test (actual tokens: {prompt_tokens}):")
        print(f"    Expected: {expected}")
        print(f"    Got:      {output[:200]}")
        print(f"    PASS: {needle_pass}")

        # Test 2: Long reasoning chain
        prompt2, expected2 = build_long_reasoning_prompt(ctx_len)
        output2, usage2 = query_server(prompt2, args.port, args.max_tokens)
        prompt_tokens2 = usage2.get("prompt_tokens", "?")
        reasoning_pass = expected2 in output2
        print(f"  Reasoning test (actual tokens: {prompt_tokens2}):")
        print(f"    Expected contains: {expected2}")
        print(f"    Got:      {output2[:200]}")
        print(f"    PASS: {reasoning_pass}")

        results.append({
            "ctx_len": ctx_len,
            "actual_prompt_tokens_needle": prompt_tokens,
            "actual_prompt_tokens_reasoning": prompt_tokens2,
            "needle_pass": needle_pass,
            "needle_output": output[:200],
            "reasoning_pass": reasoning_pass,
            "reasoning_output": output2[:200],
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Ctx Len':>10} | {'Needle':>8} | {'Reasoning':>10}")
    print("-" * 40)
    for r in results:
        print(f"{r['ctx_len']:>10} | {'PASS' if r['needle_pass'] else 'FAIL':>8} | "
              f"{'PASS' if r['reasoning_pass'] else 'FAIL':>10}")

    # Save results
    out_file = f"fp8_kv_test_port{args.port}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {out_file}")


if __name__ == "__main__":
    main()
