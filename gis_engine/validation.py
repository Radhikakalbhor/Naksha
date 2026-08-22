# ============================================================
# NAKSHA - ACCURACY & VALIDATION UTILITIES (Day 11)
# ============================================================

from typing import Dict, Any, List
import numpy as np
from shapely.geometry import shape, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from gis_engine.topology.geometry import validate_geometry


def compute_vector_metrics(pred_geoms: List[BaseGeometry], gt_geoms: List[BaseGeometry], iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate Precision, Recall, F1 score, and mean IoU for vector features.
    """
    valid_preds = [g for g in (validate_geometry(g) for g in pred_geoms) if g is not None]
    valid_gts = [g for g in (validate_geometry(g) for g in gt_geoms) if g is not None]

    if not valid_preds and not valid_gts:
        return {"iou": 1.0, "precision": 1.0, "recall": 1.0, "f1_score": 1.0, "tp": 0, "fp": 0, "fn": 0}

    if not valid_preds:
        return {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "tp": 0, "fp": 0, "fn": len(valid_gts)}

    if not valid_gts:
        return {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "tp": 0, "fp": len(valid_preds), "fn": 0}

    matched_gt = set()
    tp = 0
    fp = 0
    ious = []

    for pred in valid_preds:
        best_iou = 0.0
        best_gt_idx = -1
        for idx, gt in enumerate(valid_gts):
            if idx in matched_gt:
                continue
            try:
                intersection = pred.intersection(gt).area
                union = pred.union(gt).area
                if union > 0:
                    iou = intersection / union
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx
            except Exception:
                continue

        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
            ious.append(best_iou)
        else:
            fp += 1

    fn = len(valid_gts) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0

    return {
        "iou": round(mean_iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn
    }
