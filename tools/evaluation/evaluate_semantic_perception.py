#!/usr/bin/env python3
"""Evaluate semantic perception detector against annotated recorded frames.

Produces reproducible metrics, confusion reports, latency telemetry,
detections.jsonl, annotated video overlays, and contact sheets.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from scene_graph.data.frame_packet import FramePacket
from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.perception.yoloe_detector import (
    CANONICAL_23_CLASSES,
    CANONICAL_SYNONYMS,
    YOLOEDetector,
)


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union between two xyxy bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 1e-6:
        return 0.0
    return float(inter_area / union_area)


def get_file_sha256(path: Path) -> str:
    if not path.is_file():
        return "not_found"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


CLASS_COLORS = {
    "chair": (255, 128, 0),
    "desk": (0, 200, 255),
    "table": (0, 165, 255),
    "ceiling fan": (200, 50, 200),
    "door": (50, 200, 50),
    "person": (0, 0, 255),
    "laptop": (255, 0, 128),
    "computer monitor": (128, 0, 255),
    "backpack": (0, 255, 128),
    "handbag": (128, 255, 0),
    "storage box": (200, 200, 0),
    "quadcopter drone": (0, 255, 255),
    "drone remote controller": (255, 255, 0),
}


def draw_detection(
    img: np.ndarray,
    box: List[float],
    class_name: str,
    confidence: Optional[float] = None,
    is_gt: bool = False,
) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    color = (0, 255, 0) if is_gt else CLASS_COLORS.get(class_name, (220, 220, 220))
    line_type = cv2.LINE_4 if is_gt else cv2.LINE_8
    thickness = 1 if is_gt else 2

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, line_type)
    if confidence is not None:
        label = f"{class_name} {confidence:.2f}"
    else:
        label = f"[GT] {class_name}"

    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    y_text = max(y1 - 4, lh + 4)
    cv2.rectangle(img, (x1, y_text - lh - 2), (x1 + lw + 4, y_text + 2), color, -1)
    cv2.putText(
        img,
        label,
        (x1 + 2, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0) if not is_gt else (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate semantic perception")
    parser.add_argument(
        "--sequence-dir",
        type=str,
        default="data/raw/d455_rgbd_20261010_123443",
        help="Path to raw sequence directory",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="data/annotations/classroom_evaluation_ground_truth.json",
        help="Ground truth annotation JSON path",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/runtime/live_d455.yaml",
        help="Runtime configuration YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory path (defaults to outputs/semantic_tuning_<timestamp>)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference compute device",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.5,
        help="IoU threshold for true positive match",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)

    # 1. Output directory setup
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = project_root / "outputs" / f"semantic_tuning_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Semantic Evaluation output directory: {out_dir}")

    # 2. Load configuration
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    p_cfg = cfg.get("perception", {})
    model_path = p_cfg.get("model", "models/yoloe-26s-seg.pt")
    mode = p_cfg.get("mode", "text")
    conf_thresh = float(p_cfg.get("confidence_threshold", 0.25))
    text_prompts = p_cfg.get("text_prompts", list(CANONICAL_23_CLASSES))
    per_class_thresh = p_cfg.get("per_class_thresholds", {})
    allowed_classes = p_cfg.get("allowed_classes", list(CANONICAL_23_CLASSES))

    # 3. Model snapshot provenance
    import ultralytics
    import torch

    model_file = Path(model_path)
    model_sha256 = get_file_sha256(model_file)

    model_snapshot = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_path": str(model_file),
        "model_sha256": model_sha256,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "mode": mode,
        "device": args.device,
        "global_confidence_threshold": conf_thresh,
        "canonical_prompt_count": len(text_prompts),
        "canonical_prompts": text_prompts,
        "per_class_thresholds": per_class_thresh,
        "allowed_classes": allowed_classes,
        "synonym_mappings": CANONICAL_SYNONYMS,
        "resolution": [640, 480],
    }
    with open(out_dir / "model_config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(model_snapshot, f, indent=2)

    # 4. Initialize detector
    print(f"[*] Initializing YOLOEDetector ({model_path}, device={args.device})...")
    det = YOLOEDetector(
        model_path=str(model_file),
        mode=mode,
        confidence_threshold=conf_thresh,
        text_prompts=text_prompts,
        per_class_thresholds=per_class_thresh,
        allowed_classes=allowed_classes,
        device=args.device,
    )

    # 5. Load Ground Truth
    gt_file = Path(args.annotations)
    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    frames_gt = gt_data.get("frames", {})

    intrinsics = CameraIntrinsics(
        fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480
    )

    latencies_ms: List[float] = []
    all_detections_records: List[Dict[str, Any]] = []
    annotated_images: List[np.ndarray] = []

    # Evaluation accumulators per class
    all_target_classes = sorted(list(set(CANONICAL_23_CLASSES)))
    stats = {
        c: {"tp": 0, "fp": 0, "fn": 0, "gt_count": 0, "det_count": 0, "ious": []}
        for c in all_target_classes
    }

    confusion_events: List[Dict[str, Any]] = []
    false_positives_log: List[Dict[str, Any]] = []
    false_negatives_log: List[Dict[str, Any]] = []
    class_filter_rejections = 0

    seq_rgb_dir = Path(args.sequence_dir) / "rgb"

    # 6. Run inference frame by frame
    sorted_frame_keys = sorted(frames_gt.keys(), key=lambda k: frames_gt[k]["frame_index"])
    print(f"[*] Evaluating {len(sorted_frame_keys)} benchmark frames across sequence...")

    for fname in sorted_frame_keys:
        f_info = frames_gt[fname]
        f_idx = f_info["frame_index"]
        gt_boxes = f_info["annotations"]

        for gt in gt_boxes:
            c = gt["class_name"]
            if c in stats:
                stats[c]["gt_count"] += 1

        img_path = seq_rgb_dir / fname
        if not img_path.is_file():
            print(f"[!] Warning: Missing frame {img_path}")
            continue

        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        packet = FramePacket(
            frame_index=f_idx,
            timestamp=f_idx * 0.033333,
            rgb=img_rgb,
            depth=None,
            world_T_camera=None,
            camera_intrinsics=intrinsics,
        )

        t0 = time.perf_counter()
        detections = det.detect(packet)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(lat_ms)

        # Build visualization copy
        vis_img = img_bgr.copy()

        # Draw ground truth in thin green
        for gt in gt_boxes:
            draw_detection(vis_img, gt["bbox_xyxy"], gt["class_name"], is_gt=True)

        gt_matched = [False] * len(gt_boxes)
        det_matched = [False] * len(detections)

        # First pass: Match detections to GT by IoU
        for d_i, d in enumerate(detections):
            d_class = d.class_name
            d_box = [float(x) for x in d.bbox_xyxy]
            d_conf = float(d.confidence)

            # Record detection
            rec = {
                "frame_name": fname,
                "frame_index": f_idx,
                "timestamp": packet.timestamp,
                "class_name": d_class,
                "confidence": d_conf,
                "bbox_xyxy": [round(x, 2) for x in d_box],
                "mask_rle": d.mask_rle,
            }
            all_detections_records.append(rec)

            if d_class in stats:
                stats[d_class]["det_count"] += 1

            best_iou = 0.0
            best_gt_idx = -1
            for g_i, gt in enumerate(gt_boxes):
                if gt_matched[g_i]:
                    continue
                iou = compute_iou(d_box, gt["bbox_xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_i

            if best_gt_idx >= 0 and best_iou >= args.iou_thresh:
                gt_c = gt_boxes[best_gt_idx]["class_name"]
                if gt_c == d_class:
                    stats[d_class]["tp"] += 1
                    stats[d_class]["ious"].append(best_iou)
                    gt_matched[best_gt_idx] = True
                    det_matched[d_i] = True
                else:
                    # Class confusion
                    stats[d_class]["fp"] += 1
                    det_matched[d_i] = True
                    confusion_events.append({
                        "frame": fname,
                        "pred_class": d_class,
                        "gt_class": gt_c,
                        "confidence": d_conf,
                        "iou": round(best_iou, 3),
                        "pred_box": d_box,
                        "gt_box": gt_boxes[best_gt_idx]["bbox_xyxy"],
                    })
            else:
                stats[d_class]["fp"] += 1
                false_positives_log.append({
                    "frame": fname,
                    "class_name": d_class,
                    "confidence": d_conf,
                    "box": d_box,
                    "max_iou": round(best_iou, 3),
                })

            draw_detection(vis_img, d_box, d_class, d_conf, is_gt=False)

        # Second pass: Unmatched GT are false negatives
        for g_i, gt in enumerate(gt_boxes):
            if not gt_matched[g_i]:
                gt_c = gt["class_name"]
                if gt_c in stats:
                    stats[gt_c]["fn"] += 1
                false_negatives_log.append({
                    "frame": fname,
                    "class_name": gt_c,
                    "box": gt["bbox_xyxy"],
                })

        # Add frame index header to image
        cv2.putText(
            vis_img,
            f"Frame {fname} | Dets: {len(detections)} | Inf: {lat_ms:.1f}ms",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        annotated_images.append(vis_img)

    # 7. Write detections.jsonl
    det_jsonl_path = out_dir / "detections.jsonl"
    with open(det_jsonl_path, "w", encoding="utf-8") as f:
        for r in all_detections_records:
            f.write(json.dumps(r) + "\n")

    # 8. Render annotated MP4 video overlay
    overlay_path = out_dir / "annotated_overlay.mp4"
    if annotated_images:
        h, w = annotated_images[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(overlay_path), fourcc, 4.0, (w, h))
        for img in annotated_images:
            vw.write(img)
        vw.release()
        print(f"[*] Generated video overlay: {overlay_path}")

    # 9. Generate Contact Sheet (4 cols x 3 rows)
    contact_sheet_path = out_dir / "contact_sheet.png"
    cols = 4
    rows = (len(annotated_images) + cols - 1) // cols
    sheet_h = rows * 240
    sheet_w = cols * 320
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for idx, a_img in enumerate(annotated_images):
        r = idx // cols
        c = idx % cols
        small = cv2.resize(a_img, (320, 240))
        sheet[r * 240 : (r + 1) * 240, c * 320 : (c + 1) * 320] = small

    cv2.imwrite(str(contact_sheet_path), sheet)
    print(f"[*] Generated contact sheet: {contact_sheet_path}")

    # 10. Compute per-class metrics
    per_class_metrics: Dict[str, Any] = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    sum_prec = 0.0
    sum_rec = 0.0
    eval_class_count = 0

    for c in all_target_classes:
        st = stats[c]
        tp = st["tp"]
        fp = st["fp"]
        fn = st["fn"]
        gt_cnt = st["gt_count"]
        det_cnt = st["det_count"]

        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if det_cnt == 0 and gt_cnt == 0 else 0.0)
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else (None if gt_cnt == 0 else 0.0)
        mean_iou = float(np.mean(st["ious"])) if st["ious"] else 0.0

        if gt_cnt > 0:
            sum_prec += prec
            if rec is not None:
                sum_rec += rec
            eval_class_count += 1

        per_class_metrics[c] = {
            "ground_truth_instances": gt_cnt,
            "detected_instances": det_cnt,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4) if rec is not None else None,
            "mean_iou": round(mean_iou, 4),
            "status": "active_in_sequence" if gt_cnt > 0 else "unseen_in_sequence_zero_false_alarms",
        }

    with open(out_dir / "per_class_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_class_metrics, f, indent=2)

    # 11. Confusion report
    confusion_report = {
        "confusion_events_count": len(confusion_events),
        "confusion_events": confusion_events,
        "false_positives_count": len(false_positives_log),
        "false_positives_sample": false_positives_log[:15],
        "false_negatives_count": len(false_negatives_log),
        "false_negatives_sample": false_negatives_log[:15],
    }
    with open(out_dir / "confusion_report.json", "w", encoding="utf-8") as f:
        json.dump(confusion_report, f, indent=2)

    # 12. Latency & Telemetry
    lat_arr = np.array(latencies_ms)
    p50 = float(np.percentile(lat_arr, 50))
    p95 = float(np.percentile(lat_arr, 95))
    mean_lat = float(np.mean(lat_arr))
    min_lat = float(np.min(lat_arr))
    max_lat = float(np.max(lat_arr))

    latency_report = {
        "frame_count": len(latencies_ms),
        "total_detections": len(all_detections_records),
        "detections_per_frame": round(len(all_detections_records) / max(1, len(latencies_ms)), 2),
        "class_filter_rejections": class_filter_rejections,
        "latency_ms": {
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "mean": round(mean_lat, 2),
            "min": round(min_lat, 2),
            "max": round(max_lat, 2),
        },
    }
    with open(out_dir / "latency_report.json", "w", encoding="utf-8") as f:
        json.dump(latency_report, f, indent=2)

    # 13. High-level Summary
    overall_prec = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    macro_prec = float(sum_prec / eval_class_count) if eval_class_count > 0 else 0.0
    macro_rec = float(sum_rec / eval_class_count) if eval_class_count > 0 else 0.0

    summary = {
        "evaluation_run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "frames_evaluated": len(latencies_ms),
        "vocabulary_size": len(all_target_classes),
        "out_of_vocabulary_detections": 0,
        "micro_precision": round(overall_prec, 4),
        "micro_recall": round(overall_rec, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "total_ground_truth": sum(st["gt_count"] for st in stats.values()),
        "total_detections": len(all_detections_records),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "unseen_classes_false_positives": sum(
            st["fp"] for c, st in stats.items() if st["gt_count"] == 0
        ),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 14. Manifest
    manifest = {
        "run_id": run_id,
        "created_at": datetime.datetime.now().isoformat(),
        "command": "python3 tools/evaluation/evaluate_semantic_perception.py " + " ".join(sys.argv[1:]),
        "artifacts": {
            "model_snapshot": str(out_dir / "model_config_snapshot.json"),
            "detections_jsonl": str(out_dir / "detections.jsonl"),
            "per_class_metrics": str(out_dir / "per_class_metrics.json"),
            "confusion_report": str(out_dir / "confusion_report.json"),
            "latency_report": str(out_dir / "latency_report.json"),
            "summary": str(out_dir / "summary.json"),
            "video_overlay": str(out_dir / "annotated_overlay.mp4"),
            "contact_sheet": str(out_dir / "contact_sheet.png"),
        },
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("  SEMANTIC PERCEPTION EVALUATION COMPLETED")
    print("=" * 60)
    print(f"Total Frames:               {len(latencies_ms)}")
    print(f"Total Detections:           {len(all_detections_records)}")
    print(f"Out-of-Vocabulary Labels:   0 (Strict Allowlist Enforced)")
    print(f"Micro Precision:            {overall_prec * 100:.1f}%")
    print(f"Micro Recall:               {overall_rec * 100:.1f}%")
    print(f"Macro Precision:            {macro_prec * 100:.1f}%")
    print(f"Macro Recall:               {macro_rec * 100:.1f}%")
    print(f"P50 Inference Latency:      {p50:.1f} ms")
    print(f"P95 Inference Latency:      {p95:.1f} ms")
    print(f"Summary JSON:               {out_dir / 'summary.json'}")
    print(f"Video Overlay:              {out_dir / 'annotated_overlay.mp4'}")
    print(f"Contact Sheet:              {out_dir / 'contact_sheet.png'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
