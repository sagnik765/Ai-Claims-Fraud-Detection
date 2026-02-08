from __future__ import annotations

import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="score", choices=["train", "score"])
    parser.add_argument("--data", default="data/sample/claims.jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cmd = ["python", "-m", "src.main", "--mode", args.mode, "--data", args.data]
    if args.output:
        cmd += ["--output", args.output]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
