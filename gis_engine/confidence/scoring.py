def compute_confidence(
    prediction_mean: float | None = None,
    prediction_min: float | None = None,
    prediction_max: float | None = None,
) -> float:
    """
    Confidence scoring stub for GIS Engine.

    Day 5 requires a confidence score to be associated
    with each vector feature. The roadmap does not define
    the final production scoring formula, so this function
    provides a deterministic placeholder.
    """

    if prediction_mean is None:
        return 0.0

    confidence = float(prediction_mean)

    if prediction_min is not None:
        confidence = max(
            confidence,
            float(prediction_min)
        )

    if prediction_max is not None:
        confidence = min(
            confidence,
            float(prediction_max)
        )

    return max(
        0.0,
        min(1.0, confidence)
    )