"""Isolated batch frontend for dghs-imgutils anime face detection."""

import argparse
import json
from pathlib import Path

from imgutils.detect import detect_faces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--level", default="n")
    parser.add_argument("--version", default="v1.4")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.60)
    args = parser.parse_args()

    paths = json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = {}
    for raw_path in paths:
        try:
            detections = detect_faces(
                raw_path,
                level=args.level,
                version=args.version,
                conf_threshold=args.confidence,
                iou_threshold=args.iou,
            )
            results[raw_path] = {
                "detections": [
                    {
                        "bbox": [int(value) for value in bbox],
                        "label": str(label),
                        "confidence": float(confidence),
                    }
                    for bbox, label, confidence in detections
                ]
            }
        except Exception as exc:
            results[raw_path] = {"error": f"{type(exc).__name__}: {exc}"}
    Path(args.output).write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
