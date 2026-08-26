"""
Run this first to show exactly what UAH-DriveSet looks like on disk.
python diagnose.py --data_dir data/raw
"""
import argparse
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw")
    args = parser.parse_args()

    root = Path(args.data_dir)
    if not root.exists():
        print(f"ERROR: {root} does not exist")
        return

    print(f"\n{'='*60}")
    print(f"  Directory tree under: {root.resolve()}")
    print(f"{'='*60}")

    all_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        depth = len(Path(dirpath).relative_to(root).parts)
        indent = "  " * depth
        rel = Path(dirpath).relative_to(root)
        if depth > 0:
            print(f"{indent}{rel.name}/")
        for fname in sorted(filenames):
            fpath = Path(dirpath) / fname
            all_files.append(fpath)
            print(f"{'  '*(depth+1)}{fname}")

    print(f"\nTotal files found: {len(all_files)}")

    if not all_files:
        print("\nNo files found — check that you extracted the zip into data/raw/")
        return

    txt_files = [f for f in all_files if f.suffix.lower() == ".txt"]
    print(f"  .txt files: {len(txt_files)}")

    if txt_files:
        sample = txt_files[0]
        print(f"\n{'='*60}")
        print(f"  First 5 lines of: {sample.name}")
        print(f"{'='*60}")
        with open(sample, errors="replace") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(repr(line.rstrip()))

        print(f"\n  Column count in first data line: ", end="")
        with open(sample, errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    parts = stripped.split()
                    print(len(parts), "->", parts[:5], "...")
                    break

if __name__ == "__main__":
    main()
