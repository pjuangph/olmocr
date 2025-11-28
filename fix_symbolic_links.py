#!/usr/bin/env python3
import os
import shutil
import tempfile
from collections import defaultdict

ROOT = "data"  # adjust if needed

# 1) Build an index of basename -> full paths (non-symlinks only).
index = defaultdict(list)
for dirpath, _, filenames in os.walk(ROOT):
    for name in filenames:
        full = os.path.join(dirpath, name)
        if os.path.islink(full):  # skip links in the index
            continue
        index[name].append(full)

def replace_symlink(link_path: str, source_path: str) -> None:
    """Atomically replace symlink with file content."""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(link_path))
    os.close(tmp_fd)
    shutil.move(source_path, tmp_path)  # preserve mode/timestamps
    os.replace(tmp_path, link_path)

for dirpath, _, filenames in os.walk(ROOT):
    for name in filenames:
        link = os.path.join(dirpath, name)
        if not os.path.islink(link):
            continue

        real_target = os.path.realpath(link)
        if os.path.exists(real_target):
            # Symlink points to a real file
            replace_symlink(link, real_target)
            print(f"replaced (existing target): {link}")
            continue

        # Broken link: try to find by basename
        candidates = index.get(name, [])
        if len(candidates) == 1:
            replace_symlink(link, candidates[0])
            print(f"replaced (found by name): {link} -> {candidates[0]}")
        elif len(candidates) == 0:
            print(f"skip (no match found): {link} -> {real_target}")
        else:
            print(f"skip (multiple matches): {link} -> {candidates}")

print("Done.")
