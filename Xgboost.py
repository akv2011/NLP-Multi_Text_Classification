import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb
import time
from datetime import datetime


class XGBoostCybercrimeClassifier:
    def __init__(self):
        print("\n" + "=" * 50)
        print("Initializing XGBoost Cybercrime Classifier")
        print("=" * 50)
        
        self.label_encoders = {
            'category': LabelEncoder(),
            'sub_category': LabelEncoder()
        }
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            strip_accents='unicode',
            decode_error='replace'
        )
        
        # XGBoost parameters
        self.params = {
            'objective': 'multi:softprob',
            'eval_metric': ['mlogloss', 'merror'],
            'tree_method': 'hist',
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_child_weight': 2,
            'gamma': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_estimators': 200
        }
        
        print("\nModel parameters:")
        for param, value in self.params.items():
            print(f"  {param}: {value}")
        
        self.trained_models = {}
        self.class_weights = {}
    
    def compute_class_weights(self, y, label_type):
        print(f"\nComputing class weights for {label_type}...")
        unique_classes = np.unique(y)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y
        )
        self.class_weights[label_type] = dict(zip(unique_classes, weights))
        print(f"Found {len(unique_classes)} unique classes")
        print("Class distribution:")
        for cls, weight in zip(unique_classes, weights):
            count = np.sum(y == cls)
            print(f"  Class {cls}: {count} samples (weight: {weight:.2f})")
        return np.array([self.class_weights[label_type][label] for label in y])
    
    def fit(self, X, y_category, y_subcategory):
        print("\n" + "=" * 50)
        print(f"Starting model training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
        start_time = time.time()
        
        print("\nGenerating TF-IDF features...")
        X_tfidf = self.tfidf.fit_transform(X)
        print(f"TF-IDF matrix shape: {X_tfidf.shape}")
        print(f"Vocabulary size: {len(self.tfidf.vocabulary_)}")
        
        # Train category model
        print("\n" + "-" * 50)
        print("Training category model...")
        print("-" * 50)
        y_category_encoded = self.label_encoders['category'].fit_transform(y_category)
        category_weights = self.compute_class_weights(y_category_encoded, 'category')
        
        self.trained_models['category'] = xgb.XGBClassifier(
            **self.params,
            num_class=len(np.unique(y_category_encoded))
        )
        
        category_start = time.time()
        self.trained_models['category'].fit(
            X_tfidf, 
            y_category_encoded,
            sample_weight=category_weights,
            verbose=True,
            eval_set=[(X_tfidf, y_category_encoded)],
        )
        print(f"\nCategory model training completed in {(time.time() - category_start)/60:.1f} minutes")
        
        # Train subcategory model
        print("\n" + "-" * 50)
        print("Training subcategory model...")
        print("-" * 50)
        y_subcategory_encoded = self.label_encoders['sub_category'].fit_transform(y_subcategory)
        subcategory_weights = self.compute_class_weights(y_subcategory_encoded, 'sub_category')
        
        self.trained_models['sub_category'] = xgb.XGBClassifier(
            **self.params,
            num_class=len(np.unique(y_subcategory_encoded))
        )
        
        subcategory_start = time.time()
        self.trained_models['sub_category'].fit(
            X_tfidf, 
            y_subcategory_encoded,
            sample_weight=subcategory_weights,
            verbose=True,
            eval_set=[(X_tfidf, y_subcategory_encoded)],
        )
        print(f"\nSubcategory model training completed in {(time.time() - subcategory_start)/60:.1f} minutes")
        
        total_time = time.time() - start_time
        print("\n" + "=" * 50)
        print(f"Total training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total training time: {total_time/60:.1f} minutes")
        print("=" * 50)
        return self
    
    def predict(self, X):
        print("\n" + "=" * 50)
        print(f"Starting predictions at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
        
        start_time = time.time()
        
        print("\nTransforming texts to TF-IDF features...")
        X_tfidf = self.tfidf.transform(X)
        print(f"Feature matrix shape: {X_tfidf.shape}")
        
        print("\nMaking category predictions...")
        category_predictions = self.trained_models['category'].predict(X_tfidf)
        
        print("\nMaking subcategory predictions...")
        subcategory_predictions = self.trained_models['sub_category'].predict(X_tfidf)
        
        print(f"\nPredictions completed in {(time.time() - start_time)/60:.1f} minutes")
        
        return (
            self.label_encoders['category'].inverse_transform(category_predictions),
            self.label_encoders['sub_category'].inverse_transform(subcategory_predictions)
        )
    
    def evaluate(self, X, y_category, y_subcategory):
        print("\n" + "=" * 50)
        print("Evaluating model performance")
        print("=" * 50)
        
        # Predict
        pred_category, pred_subcategory = self.predict(X)
        
        # Category metrics
        print("\nCategory Classification Report:")
        y_category_encoded = self.label_encoders['category'].transform(y_category)
        category_f1 = f1_score(y_category_encoded, self.label_encoders['category'].transform(pred_category), average='weighted')
        print(classification_report(y_category_encoded, self.label_encoders['category'].transform(pred_category)))
        print(f"Category F1 Score (Weighted): {category_f1:.4f}")
        
        # Subcategory metrics
        print("\nSubcategory Classification Report:")
        y_subcategory_encoded = self.label_encoders['sub_category'].transform(y_subcategory)
        subcategory_f1 = f1_score(y_subcategory_encoded, self.label_encoders['sub_category'].transform(pred_subcategory), average='weighted')
        print(classification_report(y_subcategory_encoded, self.label_encoders['sub_category'].transform(pred_subcategory)))
        print(f"Subcategory F1 Score (Weighted): {subcategory_f1:.4f}")

        return category_f1, subcategory_f1

# Example usage
if __name__ == "__main__":
    print("\nLoading data...")
    train_df = pd.read_csv('Train_cleaned.csv')
    test_df = pd.read_csv('Test_cleaned.csv')
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    classifier = XGBoostCybercrimeClassifier()
    classifier.fit(train_df['crimeaditionalinfo'], train_df['category'], train_df['sub_category'])
    classifier.evaluate(test_df['crimeaditionalinfo'], test_df['category'], test_df['sub_category'])
