"""Rescore the released predictions with the original E2E metrics package."""
import gzip
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCORER = HERE.parents[1] / "_third_party/e2e-metrics/measure_scores.py"
METRICS = ("BLEU", "NIST", "METEOR", "ROUGE_L", "CIDEr")
EXPECTED = {
    0: (0.6988, 8.8159, 0.4655, 0.7112, 2.5228),
    42: (0.7001, 8.8263, 0.4665, 0.7146, 2.5186),
    110: (0.7030, 8.8523, 0.4662, 0.7203, 2.5320),
}


def main():
    bleu = []
    with tempfile.TemporaryDirectory() as directory:
        reference = Path(directory) / "references.txt"
        reference.write_bytes(gzip.decompress((HERE / "references.txt.gz").read_bytes()))
        for seed, expected in EXPECTED.items():
            output = subprocess.check_output(
                [sys.executable, str(SCORER), str(reference), str(HERE / f"seed{seed}.txt"), "-p"],
                text=True,
            )
            scores = dict(re.findall(r"^(\w+): ([\d.]+)$", output.split("SCORES:")[-1], re.M))
            values = tuple(float(scores[key]) for key in METRICS)
            print(f"Seed {seed}: {scores}", flush=True)
            if values != expected:
                raise ValueError(f"Expected {dict(zip(METRICS, expected))}")
            bleu.append(values[0] * 100)
    print(f"BLEU mean {statistics.mean(bleu):.2f}, sample SD {statistics.stdev(bleu):.2f}")


if __name__ == "__main__":
    main()
