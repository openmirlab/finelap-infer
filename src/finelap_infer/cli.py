"""Small JSON CLI over the public session; contains no evaluation harness."""

import argparse
import json

from .session import FineLAPSession


def main() -> None:
    parser = argparse.ArgumentParser(prog="finelap-infer")
    parser.add_argument("audio")
    parser.add_argument("phrases", nargs="+")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    with FineLAPSession(
        device=args.device,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    ) as session:
        result = session.infer(args.audio, args.phrases, threshold=args.threshold)
    print(json.dumps(result.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
