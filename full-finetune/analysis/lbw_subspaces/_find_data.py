import os
pact_dir = os.path.dirname(os.path.abspath(__file__))
for root, dirs, files in os.walk(pact_dir):
    for f in sorted(files):
        full = os.path.join(root, f)
        size = os.path.getsize(full)
        print(f"{size:>10d}  {full}")
