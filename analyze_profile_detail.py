import gzip
import json
import os
import sys
from collections import defaultdict

trace_dir = sys.argv[1]
trace_files = [f for f in os.listdir(trace_dir) if f.endswith('.json') or f.endswith('.json.gz')]
trace_file = os.path.join(trace_dir, trace_files[0])
print(f'Analyzing: {trace_file}')

if trace_file.endswith('.gz'):
    with gzip.open(trace_file, 'rt') as f:
        data = json.load(f)
else:
    with open(trace_file) as f:
        data = json.load(f)

events = data.get('traceEvents', [])

# Find all elementwise_kernel copy events with their call stacks
copy_events = []
for evt in events:
    if evt.get('cat') == 'kernel' and 'dur' in evt:
        name = evt['name']
        if 'elementwise_kernel<128, 4' in name and 'direct_copy' in name:
            copy_events.append(evt)

print(f'Total direct_copy elementwise_kernel calls: {len(copy_events)}')
print(f'Total time: {sum(e["dur"] for e in copy_events)/1000:.2f} ms')

# Group by duration buckets to identify different copy sizes
buckets = defaultdict(list)
for evt in copy_events:
    dur = evt['dur']
    if dur < 5:
        buckets['<5us'].append(evt)
    elif dur < 15:
        buckets['5-15us'].append(evt)
    elif dur < 30:
        buckets['15-30us'].append(evt)
    elif dur < 60:
        buckets['30-60us'].append(evt)
    elif dur < 100:
        buckets['60-100us'].append(evt)
    else:
        buckets['>100us'].append(evt)

print('\nCopy kernel duration distribution:')
for bucket in ['<5us', '5-15us', '15-30us', '30-60us', '60-100us', '>100us']:
    evts = buckets.get(bucket, [])
    if evts:
        total = sum(e['dur'] for e in evts)
        avg = total / len(evts)
        print(f'  {bucket:>10}: {len(evts):>4} calls, total={total/1000:.2f}ms, avg={avg:.1f}us')

# Look at the correlation ID to find CPU-side callers
# Try to find the python stack traces associated with copy kernels
cpu_events = {}
for evt in events:
    if evt.get('ph') == 'X' and 'cat' in evt:
        if evt['cat'] in ('cpu_op', 'python_function'):
            tid = evt.get('tid', '')
            ts = evt.get('ts', 0)
            dur = evt.get('dur', 0)
            cpu_events.setdefault(tid, []).append(evt)

# Try correlation IDs
print('\nLooking for correlation IDs on copy kernels...')
corr_ids = set()
for evt in copy_events[:5]:
    args = evt.get('args', {})
    corr = args.get('correlation', args.get('External id', None))
    if corr:
        corr_ids.add(corr)
        print(f'  Copy dur={evt["dur"]}us, correlation={corr}')

# Find CPU ops with matching correlation IDs
if corr_ids:
    print('\nMatching CPU ops:')
    for evt in events:
        if evt.get('ph') == 'X':
            args = evt.get('args', {})
            corr = args.get('correlation', args.get('External id', None))
            if corr in corr_ids:
                print(f'  {evt.get("name", "?")[:80]}  cat={evt.get("cat", "?")}  dur={evt.get("dur", "?")}us  corr={corr}')

# Count unique durations to identify distinct copy operations
from collections import Counter
dur_counter = Counter()
for evt in copy_events:
    # Round to nearest 5us
    dur_rounded = round(evt['dur'] / 5) * 5
    dur_counter[dur_rounded] += 1

print('\nCopy durations (rounded to 5us):')
for dur, count in sorted(dur_counter.items()):
    per_layer = count / 30.0
    print(f'  ~{dur:>5}us: {count:>4} calls ({per_layer:.1f} per layer)')
