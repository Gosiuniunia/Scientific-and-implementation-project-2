import joblib
import numpy as np
from itertools import combinations

def load_model(model_path):
    """
    Loads a trained SVM pipeline from a file.

    Args:
        model_path (str): Path to the saved SVM pipeline (.pkl).

    Returns:
        tuple: (scaler, svc) where
            scaler: StandardScaler object from the pipeline
            svc: SVC object from the pipeline
    """
    model = joblib.load(model_path)
    scaler = model.named_steps["scaler"]
    svc = model.named_steps["svc"]
    return scaler, svc

def count_votes(votes):
    """
    Counts votes from binary one-vs-one classifiers and handles ties.

    Args:
        votes (list or array): List of vote counts for each class per sample.

    Returns:
        int or None: Index of class with maximum votes, or None in case of a tie.
    """
    for v in votes:
        max_vote = max(v)
        if np.sum(v == max_vote) > 1:
            return None
        else:
            return np.argmax(v)
        
def predict_with_voting(X_scaled, decisions, classes, pairs, threshold = 0.2):
    """
    Converts decision_function outputs into vote counts for each class.

    Args:
        X_scaled (np.array): Scaled feature vector(s).
        decisions (np.array): Output of SVC.decision_function.
        classes (np.array): Array of class labels.
        pairs (list of tuples): List of class pairs used in one-vs-one SVM.
        threshold (float, optional): Minimum absolute decision value to count as a vote.

    Returns:
        np.array: Array of shape (n_samples, n_classes) containing vote counts.
    """
    n_samples = X_scaled.shape[0]
    votes = np.zeros((n_samples, len(classes)), dtype=int) 
    for idx, (c1, c2) in enumerate(pairs):
        for i in range(n_samples):
            val = decisions[i, idx]
            if abs(val) < threshold:
                continue
            elif val > 0:
                votes[i, np.where(classes == c1)[0][0]] += 1
            else:
                votes[i, np.where(classes == c2)[0][0]] += 1
    return votes

def map_class(prediction):
    """
    Maps class index to class label.

    Args:
        prediction (int or None): Index of predicted class or None.

    Returns:
        str or None: Class label corresponding to the index, or None.
    """
    class_mapping = {0: 'autumn', 1: 'spring', 2: 'summer', 3: 'winter'}
    if prediction is None:
        return None
    return class_mapping.get(prediction, None)

def predict_class(model_path, features):
    """
    Predicts the class of a single feature vector using a one-vs-one SVM pipeline.

    Args:
        model_path (str): Path to the saved SVM pipeline (.pkl).
        features (list or np.array): Feature vector to classify.

    Returns:
        string or None: Predicted class name, or None if there is a tie.
    """
    scaler, svc = load_model(model_path)
    classes = svc.classes_
    pairs = list(combinations(classes, 2))
    X_scaled = scaler.transform([features])
    decisions = svc.decision_function(X_scaled)
    votes = predict_with_voting(X_scaled, decisions, classes, pairs)
    prediction = count_votes(votes)
    pred_class = map_class(prediction)
    return pred_class



# features = [67, 129, 129, 209, 135, 141, 82, 140, 142]
# model_path = "svc.pkl"
# print(predict_class(model_path, features))

