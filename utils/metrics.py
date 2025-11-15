from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """
    Évalue le modèle avec métriques standard.
    
    :param y_true: Labels vrais.
    :param y_pred: Prédictions.
    :param y_pred_proba: Probabilités (optionnel).
    :return: Dict de métriques.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    print(classification_report(y_true, y_pred))
    if y_pred_proba is not None:
        from sklearn.metrics import roc_auc_score
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
    
    return metrics