def accuracy(prediction, target):
    """
    Compute classification accuracy for a single prediction.

    Args:
        prediction: Model prediction.
        target: Ground-truth label.

    Returns:
        1 if the prediction is correct, otherwise 0.
    """
    return int(prediction == target)