import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import mlflow
import time
from prefect import task, flow

# Define parameter grids
param_grids = {
    'naive_bayes': [
        {
            'vectorization': [CountVectorizer()],
            'vectorization__max_features' : [1000, 1500, 2000, 5000],
            'classifier__alpha' : [1, 10]
        }
    ],
    'decision_tree': [
        {
            'vectorization': [CountVectorizer(), TfidfVectorizer()],
            'vectorization__max_features' : [1000, 1500, 2000, 5000],
            'classifier__max_depth': [None, 5, 10]
        }
    ],
    'logistic_regression': [
        {
            'vectorization': [CountVectorizer(), TfidfVectorizer()],
            'vectorization__max_features' : [1000, 1500, 2000, 5000],
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['elasticnet'],
            'classifier__l1_ratio': [0.4, 0.5, 0.6],
            'classifier__solver': ['saga'],
            'classifier__class_weight': ['balanced']
        }
    ],
    'random_forest': [
        {
            'vectorization': [CountVectorizer()],
            'vectorization__max_features' : [1000, 1500, 2000, 5000],
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 5, 10]
        }
    ],
    'svm': [
        {
            'vectorization': [CountVectorizer(), TfidfVectorizer()],
            'vectorization__max_features' : [1000, 1500, 2000, 5000],
            'classifier__kernel': ['linear', 'rbf', 'poly'],
            'classifier__C': [0.1, 1, 10],
        }
    ]
}

@task
def load_data(file_path):
    """Load the dataset."""
    df = pd.read_csv(file_path)
    return df

@task
def split_data(df, test_size=0.2, random_state=42):
    """Split the data into train and test sets."""
    # Select relevant columns
    df = df[["Review Title", "Review text", "Ratings"]]

    # Preprocess the text and assign sentiment labels
    df['Sentiment'] = df['Ratings'].apply(lambda rating: 'negative' if rating <= 2 else 'positive')

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df[['Review text', 'Sentiment']], df['Sentiment'], test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

@task
def preprocess_data(X_train, X_test):
    """Preprocess the text data."""
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        text = str(text)
        text = text.replace('READ MORE', '')
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r':\)|:\(|:\D|:\S', '', text)
        text = text.lower()
        stop_words = set(stopwords.words('english'))
        words = text.split()
        filtered_text = [word for word in words if word not in stop_words]
        filtered_text = [lemmatizer.lemmatize(word) for word in filtered_text]
        return " ".join(filtered_text)

    X_train['clean_text'] = X_train['Review text'].apply(preprocess_text)
    X_test['clean_text'] = X_test['Review text'].apply(preprocess_text)
    return X_train, X_test

@task
def train_model(X_train, y_train, algo, param_grids):
    """Train the model using GridSearchCV."""
    pipelines = {
        'naive_bayes': Pipeline([
            ('vectorization', CountVectorizer()),
            ('classifier', MultinomialNB())
        ]),
        'decision_tree': Pipeline([
            ('vectorization', CountVectorizer()),
            ('classifier', DecisionTreeClassifier())
        ]),
        'logistic_regression': Pipeline([
            ('vectorization', CountVectorizer()),
            ('classifier', LogisticRegression())
        ]),
        'random_forest': Pipeline([
            ('vectorization', CountVectorizer()),
            ('classifier', RandomForestClassifier())
        ]),
        'svm': Pipeline([
            ('vectorization', CountVectorizer()),
            ('classifier', SVC())
        ])
    }

    grid_search = GridSearchCV(estimator=pipelines[algo], param_grid=param_grids[algo], cv=5, scoring='f1', return_train_score=True, verbose=1)
    grid_search.fit(X_train['clean_text'], y_train)

    return grid_search.best_estimator_

@task
def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    y_test_pred = model.predict(X_test['clean_text'])
    test_f1 = f1_score(y_test, y_test_pred, pos_label='positive')
    classification_rep = classification_report(y_test, y_test_pred)

    return test_f1, classification_rep

@task
def log_metrics(algo, best_model, test_f1, classification_rep, X_test, y_test):
    """Log metrics and model to MLflow."""
    mlflow.set_experiment("exp-8")
    with mlflow.start_run(run_name=algo):
        mlflow.log_params(best_model.get_params())

        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, best_model.predict(X_test['clean_text'])))

        mlflow.sklearn.log_model(best_model, "model")

    print("Test F1 Score:", test_f1)
    print("Classification Report:")
    print(classification_rep)

# @task
# def load_mlflow_model(algo):
#     """Load the trained model from MLflow."""
#     client = mlflow.tracking.MlflowClient()
#     runs = client.search_runs(experiment_ids=[583061336309648736], filter_string=f"tags.mlflow.runName = '{algo}'")
#     run_id = runs[0].info.run_id
#     model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

#     return model

# @task
# def predict(model, new_data):
#     """Make predictions using the model."""
#     lemmatizer = WordNetLemmatizer()
#     new_data_clean = [preprocess_text(doc) for doc in new_data]
#     prediction = model.predict(new_data_clean)
#     return prediction

@flow(name="Sentiment Analysis")
def sentiment_analysis(file_path="data.csv", algo="decision_tree"):
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test = preprocess_data(X_train, X_test)
    best_model = train_model(X_train, y_train, algo, param_grids)
    test_f1, classification_rep = evaluate_model(best_model, X_test, y_test)
    log_metrics(algo, best_model, test_f1, classification_rep, X_test, y_test)

if __name__ == "__main__":
    sentiment_analysis.serve(
        name = "test-deployment",
        cron="10 * * * *"
    )