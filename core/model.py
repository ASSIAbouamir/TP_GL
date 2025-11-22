from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

class ClinicalModel:
    def __init__(self, model_type='rf', random_state=42):
        """
        :param model_type: 'rf' pour RandomForest, 'lr' pour LogisticRegression optimisée.
        """
        self.model_type = model_type
        self.model = None
        self.random_state = random_state
        self.best_params = None
    
    def train(self, X_train, y_train):
        """Entraîne le modèle."""
        if self.model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            self.model.fit(X_train, y_train)
        elif self.model_type == 'lr':
            # Optimizer : GridSearch pour C (régularisation)
            base_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            param_grid = {'C': [0.1, 1, 10, 100]}
            grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            print(f"Meilleurs params LR : {self.best_params}")
        else:
            raise ValueError("model_type doit être 'rf' ou 'lr'.")
    
    def predict(self, X):
        """Prédit."""
        if self.model is None:
            raise ValueError("Entraînez d'abord le modèle.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Probabilités."""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Évalue rapidement."""
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)