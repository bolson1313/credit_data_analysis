import pandas as pd
import numpy as np
from scipy import stats
from data_loader import get_display_column_name


def get_original_column_name(column_name):
    """Mapuje nazwę kolumny na jej oryginalną nazwę."""
    # Implementacja mapowania nazw kolumn
    # Na potrzeby tego przykładu, funkcja zwraca nazwę bez zmian
    return column_name


def calculate_numerical_stats(data, columns=None):
    """Oblicza statystyki dla kolumn numerycznych."""
    if data is None:
        return None

    if columns is not None:
        stats_df = data[columns].describe()
    else:
        stats_df = data.select_dtypes(include=['number']).describe()
    
    # Zmień nazwy kolumn na proponowane
    stats_df.columns = [get_display_column_name(col) for col in stats_df.columns]
    return stats_df


def calculate_categorical_stats(data, columns=None):
    """Oblicza statystyki dla kolumn kategorycznych."""
    if data is None:
        return None

    if columns is not None:
        # Dodać mapowanie nazw kolumn
        columns = [get_original_column_name(col) for col in columns]

    if columns is None:
        # Identyfikacja kolumn kategorycznych
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        categorical_cols = [col for col in columns if col in data.columns and (
                    pd.api.types.is_object_dtype(data[col]) or pd.api.types.is_categorical_dtype(data[col]))]

    if not categorical_cols:
        return None

    results = {}
    for col in categorical_cols:
        display_name = get_display_column_name(col)
        mode_result = data[col].mode()
        mode_value = mode_result[0] if not mode_result.empty else None

        results[display_name] = {
            'Unikalne wartości': data[col].nunique(),
            'Moda': mode_value,
            'Liczba wystąpień mody': data[col].value_counts().iloc[0] if not data[col].value_counts().empty else None,
            'Brakujące wartości': data[col].isna().sum(),
            'Najczęstsze 5 wartości': data[col].value_counts().head(5).to_dict()
        }

    return results


def calculate_correlations(data, method='pearson'):
    """Oblicza korelacje między atrybutami."""
    if data is None:
        return None

    # Tylko kolumny numeryczne mogą być użyte do obliczeń korelacji
    numeric_data = data.select_dtypes(include=['number'])

    if numeric_data.empty:
        return None

    # Dostępne metody korelacji: 'pearson', 'kendall', 'spearman'
    if method not in ['pearson', 'kendall', 'spearman']:
        method = 'pearson'

    correlation_matrix = numeric_data.corr(method=method)
    return correlation_matrix