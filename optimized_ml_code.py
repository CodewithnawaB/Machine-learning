# OPTIMIZED VERSION - FAST TRAINING
# Key changes:
# 1. Reduced max_features to 1000 (from 5000)
# 2. Added max_iter limits to all classifiers
# 3. Reduced cv folds to 3 (from 5)
# 4. Made SVM train faster with LinearSVC

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC  # MUCH faster than SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV  # For LinearSVC probabilities

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_analyze_data(file_path):
    """Load dataset and perform preliminary analysis"""
    print("\n" + "="*80)
    print("Q-1: DATA LOADING AND PRELIMINARY ANALYSIS")
    print("="*80)
    
    df = pd.read_csv(file_path, encoding='utf-8')
    
    print("\n1a. First 5 rows of the dataset:")
    print(df.head())
    
    print("\n1b. Column names and data types:")
    print(df.dtypes)
    print(f"\nDataset Shape: {df.shape}")
    
    # Find sentiment and review columns
    sentiment_col = None
    review_col = None
    
    for col in df.columns:
        if 'sentiment' in col.lower() or 'label' in col.lower():
            sentiment_col = col
        if 'review' in col.lower() or 'text' in col.lower():
            review_col = col
    
    if sentiment_col is None:
        sentiment_col = df.columns[1]
    if review_col is None:
        review_col = df.columns[0]
    
    df = df.rename(columns={sentiment_col: 'sentiment', review_col: 'review'})
    
    # Handle missing values
    print("\n2a. Checking for null values:")
    print(df.isnull().sum())
    df = df.dropna()
    
    # Class distribution
    print("\n2b. Class Distribution:")
    class_dist = df['sentiment'].value_counts()
    print(class_dist)
    
    plt.figure(figsize=(8, 5))
    class_dist.plot(kind='bar', color=['green', 'red'])
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return df

def generate_wordcloud(df, text_column='review'):
    """Generate word cloud from Urdu reviews"""
    print("\n" + "="*80)
    print("Q-2: WORD CLOUD VISUALIZATION")
    print("="*80)
    
    all_text = ' '.join(df[text_column].astype(str))
    
    wordcloud = WordCloud(
        width=1200, 
        height=600,
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate(all_text)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud - Urdu Movie Reviews', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return wordcloud

def preprocess_urdu_text(text):
    """Comprehensive preprocessing for Urdu text"""
    text = str(text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    urdu_stopwords = {
        'کا', 'کی', 'کے', 'نے', 'ہے', 'ہیں', 'تھا', 'تھی', 'تھے', 
        'ہو', 'گا', 'گی', 'گے', 'یہ', 'وہ', 'اور', 'میں', 'سے',
        'پر', 'کو', 'کر', 'لیے', 'دیا', 'ایک', 'ہی', 'بھی', 'تو'
    }
    tokens = [word for word in text.split() if word not in urdu_stopwords]
    
    return ' '.join(tokens)

def apply_preprocessing(df, text_column='review'):
    """Apply preprocessing to entire dataset"""
    print("\n" + "="*80)
    print("Q-3: DATA PREPROCESSING")
    print("="*80)
    
    print("\nApplying preprocessing steps...")
    df['cleaned_review'] = df[text_column].apply(preprocess_urdu_text)
    
    print(f"\nExample:")
    print(f"Original: {df[text_column].iloc[0][:100]}...")
    print(f"Cleaned:  {df['cleaned_review'].iloc[0][:100]}...")
    
    return df

def extract_features(df, text_column='cleaned_review'):
    """Extract features with REDUCED dimensions for faster training"""
    print("\n" + "="*80)
    print("Q-4: FEATURE EXTRACTION (OPTIMIZED)")
    print("="*80)
    
    # CRITICAL FIX: Reduced max_features from 5000 to 1000
    vec_all = TfidfVectorizer(
        ngram_range=(1, 2),  # Only unigrams + bigrams (not trigrams)
        max_features=1000,    # REDUCED from 5000
        min_df=2,             # Ignore rare terms
        max_df=0.95           # Ignore too common terms
    )
    
    X = vec_all.fit_transform(df[text_column])
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Sparsity: {(1.0 - X.nnz / (X.shape[0] * X.shape[1])) * 100:.2f}%")
    
    return X, vec_all

def train_and_evaluate_models(X, y):
    """Train models with OPTIMIZED settings for speed"""
    print("\n" + "="*80)
    print("Q-5: CLASSIFICATION (FAST MODE)")
    print("="*80)
    
    # CRITICAL FIXES: Added max_iter and other speed optimizations
    classifiers = {
        'Naive Bayes': MultinomialNB(),
        
        'SVM': LinearSVC(  # MUCH faster than SVC
            max_iter=1000,  # CRITICAL: Prevents infinite training
            random_state=42,
            dual=False,     # Faster for n_samples > n_features
            C=1.0
        ),
        
        'Decision Tree': DecisionTreeClassifier(
            random_state=42,
            max_depth=10,
            min_samples_split=5  # Prevents overfitting
        ),
        
        'Random Forest': RandomForestClassifier(
            n_estimators=50,    # REDUCED from 100
            random_state=42,
            max_depth=10,
            n_jobs=-1,          # Use all CPU cores
            min_samples_split=5
        ),
        
        'k-NN': KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1  # Parallel processing
        )
    }
    
    # REDUCED from 5 to 3 folds for speed
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    results = {}
    
    print("\nTraining models (this should be MUCH faster now)...\n")
    
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        
        try:
            cv_results = cross_validate(
                clf, X, y, cv=skf,
                scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
                return_train_score=False,
                n_jobs=-1 if name != 'SVM' else 1  # SVM already parallel
            )
            
            results[name] = {
                'Accuracy': cv_results['test_accuracy'].mean(),
                'Precision': cv_results['test_precision_macro'].mean(),
                'Recall': cv_results['test_recall_macro'].mean(),
                'F1 Score': cv_results['test_f1_macro'].mean()
            }
            
            print(f"  ✓ Accuracy: {results[name]['Accuracy']:.4f}")
            print(f"  ✓ F1 Score: {results[name]['F1 Score']:.4f}\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            results[name] = {
                'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1 Score': 0
            }
    
    return results, classifiers, skf

def visualize_results(results):
    """Create bar charts comparing model performance"""
    print("\n" + "="*80)
    print("Q-6: COMPARATIVE ANALYSIS")
    print("="*80)
    
    results_df = pd.DataFrame(results).T
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']
    
    for ax, metric, color in zip(axes.flat, metrics, colors):
        results_df[metric].plot(kind='bar', ax=ax, color=color, alpha=0.8)
        ax.set_title(f'{metric} Comparison', fontweight='bold')
        ax.set_ylabel(metric)
        ax.set_xlabel('Algorithm')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(results_df.index, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    best_model = results_df['F1 Score'].idxmax()
    best_f1 = results_df['F1 Score'].max()
    
    print(f"\n✓ Best Model: {best_model} (F1: {best_f1:.4f})")
    
    return results_df

def ensemble_classification(X, y, skf):
    """Ensemble with OPTIMIZED base classifiers"""
    print("\n" + "="*80)
    print("Q-7: ENSEMBLE CLASSIFICATION (OPTIMIZED)")
    print("="*80)
    
    print("\nUsing HARD VOTING (faster than soft voting)")
    print("Reason: LinearSVC doesn't support probabilities directly")
    print("Hard voting: Each classifier votes for a class, majority wins\n")
    
    ensemble = VotingClassifier(
        estimators=[
            ('nb', MultinomialNB()),
            ('svm', LinearSVC(max_iter=1000, dual=False, random_state=42)),
            ('rf', RandomForestClassifier(
                n_estimators=50, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ))
        ],
        voting='hard',  # CHANGED from 'soft' to 'hard'
        n_jobs=1  # CRITICAL: Set to 1 to avoid pickle issues with voting
    )
    
    print("Training ensemble (this may take 2-5 minutes)...")
    
    try:
        cv_results = cross_validate(
            ensemble, X, y, cv=skf,
            scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
            return_train_score=False,
            verbose=1  # Show progress
        )
        
        ensemble_results = {
            'Accuracy': cv_results['test_accuracy'].mean(),
            'Precision': cv_results['test_precision_macro'].mean(),
            'Recall': cv_results['test_recall_macro'].mean(),
            'F1 Score': cv_results['test_f1_macro'].mean()
        }
        
        print("\n✓ Ensemble Performance:")
        for metric, value in ensemble_results.items():
            print(f"  {metric}: {value:.4f}")
            
    except Exception as e:
        print(f"\n✗ Ensemble failed: {e}")
        print("Trying simplified ensemble without parallel processing...")
        
        # Fallback: Train without cross-validation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        ensemble_results = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='macro'),
            'Recall': recall_score(y_test, y_pred, average='macro'),
            'F1 Score': f1_score(y_test, y_pred, average='macro')
        }
        
        print("\n✓ Ensemble Performance (single split):")
        for metric, value in ensemble_results.items():
            print(f"  {metric}: {value:.4f}")
    
    return ensemble_results

def main(file_path):
    """Execute complete ML pipeline - OPTIMIZED VERSION"""
    print("="*80)
    print("URDU MOVIE REVIEWS CLASSIFICATION - OPTIMIZED FOR SPEED")
    print("="*80)
    
    import time
    start_time = time.time()
    
    # Q1-3: Load, visualize, preprocess
    df = load_and_analyze_data(file_path)
    generate_wordcloud(df)
    df = apply_preprocessing(df)
    
    # Q4: Extract features (REDUCED dimensions)
    X, vectorizer = extract_features(df)
    y = df['sentiment']
    
    print(f"\n⚡ Training on {X.shape[0]} samples with {X.shape[1]} features...")
    
    # Q5: Train models (OPTIMIZED)
    results, classifiers, skf = train_and_evaluate_models(X, y)
    
    # Q6: Visualize
    results_df = visualize_results(results)
    
    # Q7: Ensemble (OPTIMIZED)
    ensemble_results = ensemble_classification(X, y, skf)
    
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print(f"✓ COMPLETED IN {elapsed:.1f} SECONDS!")
    print("="*80)
    
    return df, results_df, ensemble_results

# RUN IT
if __name__ == "__main__":
    FILE_PATH = r'C:\ishfaq mwt material\5_th semester DR_ATIF_KHAN_sir materail\ml_assignment_02\imdb_urdu_reviews_train.csv'
    
    df, results, ensemble_results = main(FILE_PATH)
    print("\n✓ Results saved in: df, results, ensemble_results")
