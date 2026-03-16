import gzip
import json
import os
import sys
from collections import defaultdict

trace_dir = sys.argv[1]
trace_files = [f for f in os.listdir(trace_dir) if f.endswith('.json') or f.endswith('.json.gz')]
if not trace_files:
    print('No trace files found')
    sys.exit(1)
trace_file = os.path.join(trace_dir, trace_files[0])
print(f'Analyzing: {trace_file}')

if trace_file.endswith('.gz'):
    with gzip.open(trace_file, 'rt') as f:
        data = json.load(f)
else:
    with open(trace_file) as f:
        data = json.load(f)

events = data.get('traceEvents', [])

kernel_times = defaultdict(lambda: {'count': 0, 'total_us': 0})
for evt in events:
    if evt.get('cat') == 'kernel' and 'dur' in evt:
        name = evt['name']
        short = name.split('(')[0] if '(' in name else name
        if len(short) > 80:
            short = short[:80]
        kernel_times[short]['count'] += 1
        kernel_times[short]['total_us'] += evt['dur']

sorted_kernels = sorted(kernel_times.items(), key=lambda x: -x[1]['total_us'])

print('\nTop 30 GPU kernels by total time:')
header = '{:<82} {:>6} {:>10} {:>10}'.format('Kernel', 'Count', 'Total(ms)', 'Avg(us)')
print(header)
print('-' * 115)
for name, info in sorted_kernels[:30]:
    line = '{:<82} {:>6} {:>10.2f} {:>10.1f}'.format(
        name, info['count'], info['total_us']/1000, info['total_us']/info['count'])
    print(line)

print('\n=== Copy kernels (elementwise_kernel / direct_copy) ===')
for name, info in sorted_kernels:
    if 'copy' in name.lower() or 'elementwise' in name.lower():
        print('  {:>4}x  {:>8.2f} ms  avg={:>8.1f} us  {}'.format(
            info['count'], info['total_us']/1000, info['total_us']/info['count'], name[:100]))

print('\n=== GDN chunk/l2norm kernels ===')
for name, info in sorted_kernels:
    if any(k in name.lower() for k in ['chunk', 'l2norm', 'recompute', 'fused_recurrent', 'solve_tril', 'cumsum']):
        print('  {:>4}x  {:>8.2f} ms  avg={:>8.1f} us  {}'.format(
            info['count'], info['total_us']/1000, info['total_us']/info['count'], name[:100]))
