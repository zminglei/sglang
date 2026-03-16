#!/bin/bash
set -e

source /home/jobuser/zminglei/sglang/venv/bin/activate
cd /home/jobuser/zminglei/sglang

RESULTS_FILE="/tmp/ab_bench_results.txt"
> $RESULTS_FILE

run_bench() {
    local branch=$1
    local bench_name=$2
    local bench_cmd=$3

    echo "=== [$branch] $bench_name ===" | tee -a $RESULTS_FILE

    # Kill existing server
    tmux kill-session -t server 2>/dev/null || true
    sleep 3

    # Switch branch
    if [ "$branch" = "main" ]; then
        git checkout main 2>/dev/null
        git checkout -- . 2>/dev/null
    else
        git checkout gdn-zero-copy-contiguous 2>/dev/null
        git reset --hard origin/gdn-zero-copy-contiguous 2>/dev/null
    fi

    # Start fresh server
    tmux new-session -d -s server "python -m sglang.launch_server --model-path /shared/public/elr-models/Qwen/Qwen3.5-35B-A3B/ --context-length 262144 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --port 30000 2>&1 | tee /tmp/sglang_server.log"

    # Wait for ready
    for i in $(seq 1 60); do
        sleep 3
        grep -q 'ready to roll' /tmp/sglang_server.log 2>/dev/null && break
    done

    if ! grep -q 'ready to roll' /tmp/sglang_server.log 2>/dev/null; then
        echo "ERROR: Server failed to start" | tee -a $RESULTS_FILE
        return 1
    fi

    # Run benchmark
    eval "$bench_cmd" 2>&1 | tee -a $RESULTS_FILE
    echo "" >> $RESULTS_FILE
}

# === MAIN BRANCH ===
run_bench "main" "gsm8k" \
    "python benchmark/gsm8k/bench_sglang.py --data-path /shared/public/data/gsm8k/test.jsonl --num-questions 200 --parallel 200 2>&1 | grep -E 'Accuracy|Invalid|Output throughput|Latency:'"

run_bench "main" "prefill-heavy (12k in, 64 out)" \
    "python -m sglang.bench_serving --backend sglang --num-prompts 32 --dataset-name random-ids --random-input-len 12000 --random-output-len 64 --random-range-ratio 1.0 2>&1 | grep -E 'Request throughput|Input token throughput|Output token throughput|Mean E2E|Mean TTFT|Mean TPOT'"

run_bench "main" "balanced (2k in, 2k out)" \
    "python -m sglang.bench_serving --backend sglang --num-prompts 64 --dataset-name random-ids --random-input-len 2000 --random-output-len 2000 --random-range-ratio 1.0 2>&1 | grep -E 'Request throughput|Output token throughput|Mean E2E|Mean TTFT|Mean TPOT'"

run_bench "main" "decode-heavy (128 in, 4k out)" \
    "python -m sglang.bench_serving --backend sglang --num-prompts 32 --dataset-name random-ids --random-input-len 128 --random-output-len 4000 --random-range-ratio 1.0 2>&1 | grep -E 'Request throughput|Output token throughput|Mean E2E|Mean TTFT|Mean TPOT'"

# === OPTIMIZED BRANCH ===
run_bench "optimized" "gsm8k" \
    "python benchmark/gsm8k/bench_sglang.py --data-path /shared/public/data/gsm8k/test.jsonl --num-questions 200 --parallel 200 2>&1 | grep -E 'Accuracy|Invalid|Output throughput|Latency:'"

run_bench "optimized" "prefill-heavy (12k in, 64 out)" \
    "python -m sglang.bench_serving --backend sglang --num-prompts 32 --dataset-name random-ids --random-input-len 12000 --random-output-len 64 --random-range-ratio 1.0 2>&1 | grep -E 'Request throughput|Input token throughput|Output token throughput|Mean E2E|Mean TTFT|Mean TPOT'"

run_bench "optimized" "balanced (2k in, 2k out)" \
    "python -m sglang.bench_serving --backend sglang --num-prompts 64 --dataset-name random-ids --random-input-len 2000 --random-output-len 2000 --random-range-ratio 1.0 2>&1 | grep -E 'Request throughput|Output token throughput|Mean E2E|Mean TTFT|Mean TPOT'"

run_bench "optimized" "decode-heavy (128 in, 4k out)" \
    "python -m sglang.bench_serving --backend sglang --num-prompts 32 --dataset-name random-ids --random-input-len 128 --random-output-len 4000 --random-range-ratio 1.0 2>&1 | grep -E 'Request throughput|Output token throughput|Mean E2E|Mean TTFT|Mean TPOT'"

echo "========================================" >> $RESULTS_FILE
echo "ALL BENCHMARKS COMPLETE" >> $RESULTS_FILE
echo "Results saved to $RESULTS_FILE"
cat $RESULTS_FILE
