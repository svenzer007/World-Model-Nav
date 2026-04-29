"""Compare two completed policy-eval output directories."""

from __future__ import annotations

def main() -> None:
    from policy_eval.compare_runs import main as compare_main

    compare_main()


if __name__ == "__main__":
    main()
