import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import plotly.express as px


def prepare_data_for_classification(data, target_column, test_size=0.3, random_state=42):
    """Przygotowuje dane do klasyfikacji."""
    if data is None or target_column not in data.columns:
        return None, None, None, None, None

    # Kopia danych
    df = data.copy()

    # Zakoduj zmienną docelową jeśli jest kategoryczna
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])
        target_names = list(le.classes_)  # Konwersja na listę
    else:
        unique_values = sorted(df[target_column].unique())
        target_names = [str(val) for val in unique_values]

    # Usuń wiersze z brakującymi wartościami
    df = df.dropna().reset_index(drop=True)

    # Podziel na cechy i zmienną docelową
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Wybierz tylko kolumny numeryczne
    X = X.select_dtypes(include=['number'])

    if X.empty:
        return None, None, None, None, None

    # Standardyzacja danych
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Podziel na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, target_names


def train_classification_model(X_train, y_train, model_type='logistic', **params):
    """Trenuje model klasyfikacyjny."""
    classifiers = {
        'logistic': LogisticRegression,
        'decision_tree': DecisionTreeClassifier,
        'random_forest': RandomForestClassifier,
        'svc': SVC,
        'knn': KNeighborsClassifier
    }

    classifier_class = classifiers.get(model_type)
    if classifier_class is None:
        raise ValueError(f"Nieznany typ klasyfikatora: {model_type}")

    model = classifier_class(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_classification_model(model, X_test, y_test, target_names=None):
    """Ocenia model klasyfikacyjny."""
    if model is None or X_test is None or y_test is None:
        return None

    # Predykcje
    y_pred = model.predict(X_test)

    # Jeśli nie podano nazw klas, użyj unikalnych wartości
    if target_names is None:
        target_names = [str(val) for val in sorted(np.unique(y_test))]

    # Upewnij się, że target_names jest listą stringów
    target_names = [str(name) for name in target_names]

    # Metryki
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, 
                                                    target_names=target_names, 
                                                    output_dict=True),
        'cv_scores': cross_val_score(model, X_test, y_test, cv=5),
    }
    
    metrics['cv_mean'] = metrics['cv_scores'].mean()
    metrics['cv_std'] = metrics['cv_scores'].std()
    
    return metrics


def prepare_data_for_clustering(data, columns=None, scaler='standard'):
    """Przygotowuje dane do grupowania."""
    if data is None:
        return None, None

    # Kopia danych
    df = data.copy()

    # Jeśli nie wybrano kolumn, użyj wszystkich numerycznych
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    else:
        # Filtruj tylko kolumny numeryczne
        columns = [col for col in columns if col in df.columns and 
                  pd.api.types.is_numeric_dtype(df[col])]

    if not columns:
        return None, None

    # Usuń wiersze z brakującymi wartościami
    df = df[columns].dropna().reset_index(drop=True)

    if df.empty:
        return None, None

    # Skalowanie danych
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=columns)

    return scaled_df, df


def train_kmeans_model(data, n_clusters, random_state=42):
    """Trenuje model K-means."""
    if data is None:
        return None

    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    model.fit(data)
    return model


def evaluate_clustering(data, model):
    """Ocenia wyniki grupowania."""
    if data is None or model is None:
        return None

    # Przypisanie klastrów
    labels = model.labels_

    # Podstawowe metryki
    metrics = {
        'inertia': model.inertia_,  # Suma kwadratów odległości próbek od ich najbliższego centroidu
        'silhouette': silhouette_score(data, labels),  # Współczynnik sylwetki
        'cluster_sizes': pd.Series(labels).value_counts().to_dict(),  # Rozmiary klastrów
        'labels': labels,  # Etykiety klastrów
        'centroids': model.cluster_centers_  # Centroidy klastrów
    }

    return metrics


def plot_clusters_2d(data, labels, centroids, features=None):
    """Tworzy wizualizację 2D klastrów."""
    if data is None or labels is None:
        return None

    if features is None or len(features) < 2:
        if data.shape[1] < 2:
            return None
        features = data.columns[:2].tolist()

    # Przygotuj dane do wizualizacji
    plot_data = pd.DataFrame({
        'Feature 1': data[features[0]],
        'Feature 2': data[features[1]],
        'Cluster': labels
    })

    # Stwórz wykres punktowy
    fig = px.scatter(
        plot_data, 
        x='Feature 1', 
        y='Feature 2', 
        color='Cluster',
        title=f'Grupowanie K-means: {features[0]} vs {features[1]}',
        labels={
            'Feature 1': features[0],
            'Feature 2': features[1]
        }
    )

    # Dodaj centroidy
    if centroids is not None:
        for i, centroid in enumerate(centroids):
            fig.add_scatter(
                x=[centroid[0]], 
                y=[centroid[1]],
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=12,
                    color='black',
                ),
                name=f'Centroid {i}'
            )

    return fig