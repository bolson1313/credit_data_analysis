import pandas as pd
import numpy as np
from scipy import stats


def calculate_numerical_stats(data, columns=None):
    """Oblicza statystyki dla kolumn numerycznych."""
    if data is None:
        return None

    # Wybierz kolumny numeryczne
    if columns is not None:
        # Filtruj żeby zostały tylko numeryczne z podanych kolumn
        numeric_data = data[columns].select_dtypes(include=['number'])
    else:
        numeric_data = data.select_dtypes(include=['number'])
    
    # Sprawdź czy są jakieś kolumny numeryczne
    if numeric_data.empty:
        return None
    
    # Oblicz podstawowe statystyki
    stats_df = numeric_data.describe()
    
    # USUŃ TĘ PROBLEMATYCZNĄ LINIĘ - nie zmieniaj nazw kolumn!
    # stats_df.columns = [get_display_column_name(col) for col in stats_df.columns]
    
    # Dodaj dodatkowe statystyki
    additional_stats = pd.DataFrame(index=['skewness', 'kurtosis'], columns=stats_df.columns)
    
    for col in numeric_data.columns:
        try:
            # Skośność
            additional_stats.loc['skewness', col] = numeric_data[col].skew()
            # Kurtoza
            additional_stats.loc['kurtosis', col] = numeric_data[col].kurtosis()
        except:
            additional_stats.loc['skewness', col] = np.nan
            additional_stats.loc['kurtosis', col] = np.nan
    
    # Połącz statystyki
    stats_df = pd.concat([stats_df, additional_stats])
    
    return stats_df


def calculate_categorical_stats(data, columns=None):
    """Oblicza statystyki dla kolumn kategorycznych."""
    if data is None:
        return None

    # Wybierz kolumny kategoryczne
    if columns is None:
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        categorical_cols = [col for col in columns if col in data.columns and (
                    pd.api.types.is_object_dtype(data[col]) or pd.api.types.is_categorical_dtype(data[col]))]

    if not categorical_cols:
        return None

    results = {}
    for col in categorical_cols:
        try:
            mode_result = data[col].mode()
            mode_value = mode_result[0] if not mode_result.empty else None

            results[col] = {  # Użyj oryginalnej nazwy kolumny
                'Unikalne wartości': data[col].nunique(),
                'Moda': mode_value,
                'Liczba wystąpień mody': data[col].value_counts().iloc[0] if not data[col].value_counts().empty else None,
                'Brakujące wartości': data[col].isna().sum(),
                'Najczęstsze 5 wartości': data[col].value_counts().head(5).to_dict()
            }
        except Exception as e:
            # Jeśli są problemy z kolumną, pomiń ją
            print(f"Błąd przetwarzania kolumny {col}: {e}")
            continue

    return results if results else None


def calculate_correlations(data, method='pearson'):
    """Oblicza korelacje między atrybutami."""
    if data is None:
        return None

    # Tylko kolumny numeryczne mogą być użyte do obliczeń korelacji
    numeric_data = data.select_dtypes(include=['number'])

    if numeric_data.empty or numeric_data.shape[1] < 2:
        return None

    # Dostępne metody korelacji: 'pearson', 'kendall', 'spearman'
    if method not in ['pearson', 'kendall', 'spearman']:
        method = 'pearson'

    try:
        correlation_matrix = numeric_data.corr(method=method)
        return correlation_matrix
    except Exception as e:
        print(f"Błąd obliczania korelacji: {e}")
        return None


def get_correlation_summary(correlation_matrix, threshold=0.5):
    """Zwraca podsumowanie najsilniejszych korelacji."""
    if correlation_matrix is None:
        return None
    
    # Znajdź pary z korelacją powyżej progu
    strong_correlations = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if not pd.isna(corr_val) and abs(corr_val) >= threshold:
                strong_correlations.append({
                    'Kolumna 1': correlation_matrix.columns[i],
                    'Kolumna 2': correlation_matrix.columns[j],
                    'Korelacja': corr_val,
                    'Siła': 'Silna' if abs(corr_val) >= 0.7 else 'Umiarkowana'
                })
    
    if strong_correlations:
        df = pd.DataFrame(strong_correlations)
        return df.sort_values('Korelacja', key=abs, ascending=False)
    
    return None


def detect_outliers(data, method='iqr'):
    """Wykrywa wartości odstające w danych numerycznych."""
    if data is None:
        return None
    
    numeric_data = data.select_dtypes(include=['number'])
    if numeric_data.empty:
        return None
    
    outliers_info = {}
    
    for col in numeric_data.columns:
        col_data = numeric_data[col].dropna()
        if len(col_data) == 0:
            continue
            
        if method == 'iqr':
            # Metoda IQR (Interquartile Range)
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
        elif method == 'zscore':
            # Metoda Z-score
            z_scores = np.abs(stats.zscore(col_data))
            outliers = col_data[z_scores > 3]
        
        if len(outliers) > 0:
            outliers_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(col_data)) * 100,
                'values': outliers.tolist()[:10]  # Maksymalnie 10 przykładów
            }
    
    return outliers_info if outliers_info else None