import pandas as pd
import numpy as np
import streamlit as st

def load_csv(file):
    """Wczytuje dane z pliku CSV."""
    try:
        # Reset file pointer
        file.seek(0)
        
        # Spróbuj różnych kodowań
        encodings = ['utf-8', 'latin-1', 'cp1250', 'iso-8859-1']
        data = None
        
        for encoding in encodings:
            try:
                file.seek(0)
                data = pd.read_csv(file, encoding=encoding, na_values=['?', 'NA', 'na', 'N/A', 'n/a', '', ' ', 'null', 'NULL'])
                st.success(f"✅ Plik wczytany pomyślnie (kodowanie: {encoding})")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.warning(f"Próba z kodowaniem {encoding} nie powiodła się: {str(e)}")
                continue
        
        if data is None:
            st.error("❌ Nie udało się wczytać pliku z żadnym z obsługiwanych kodowań")
            return None
        
        # Podstawowe czyszczenie danych
        data = clean_data(data)
        
        if data is None or data.empty:
            st.error("❌ Plik jest pusty lub zawiera tylko nieprawidłowe dane")
            return None
        
        # Informacje o wczytanych danych
        st.info(f"📊 Wczytano: {data.shape[0]} wierszy × {data.shape[1]} kolumn")
        
        # Sprawdź jakość danych
        missing_percent = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
        if missing_percent > 0:
            st.warning(f"⚠️ Brakujące wartości: {missing_percent:.1f}% wszystkich komórek")
        
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            st.warning(f"⚠️ Znaleziono {duplicates} zduplikowanych wierszy")
        
        return data
        
    except Exception as e:
        st.error(f"❌ Błąd wczytywania pliku: {str(e)}")
        return None

def clean_data(data):
    """Czyści wczytane dane."""
    try:
        if data is None:
            return None
        
        # Skopiuj dane
        cleaned = data.copy()
        
        # Usuń całkowicie puste wiersze i kolumny
        cleaned = cleaned.dropna(how='all').reset_index(drop=True)
        cleaned = cleaned.dropna(axis=1, how='all')
        
        # Zastąp różne reprezentacje brakujących wartości
        missing_representations = ['?', 'NA', 'na', 'N/A', 'n/a', '', ' ', 'null', 'NULL', 'None', 'NONE']
        for col in cleaned.columns:
            for missing_val in missing_representations:
                cleaned[col] = cleaned[col].replace(missing_val, np.nan)
        
        # Usuń kolumny, które są w całości puste po czyszczeniu
        cleaned = cleaned.dropna(axis=1, how='all')
        
        # Sprawdź czy zostały jakieś dane
        if cleaned.empty:
            return None
        
        return cleaned
        
    except Exception as e:
        st.error(f"❌ Błąd podczas czyszczenia danych: {str(e)}")
        return data

def get_dataset_info(data):
    """Zwraca podstawowe informacje o zbiorze danych."""
    if data is None:
        return None

    # Analiza typów danych
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    info = {
        "rows": data.shape[0],
        "columns": data.shape[1],
        "columns_names": data.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_values": data.isna().sum().sum(),
        "duplicated_rows": data.duplicated().sum(),
        "dtypes": data.dtypes,
        "memory_usage": data.memory_usage(deep=True).sum() / 1024**2  # MB
    }

    return info

def detect_column_types(data):
    """Automatycznie wykrywa i sugeruje typy kolumn."""
    if data is None:
        return {}
    
    suggestions = {}
    
    for col in data.columns:
        # Sprawdź czy kolumna wygląda na numeryczną
        try:
            pd.to_numeric(data[col], errors='raise')
            suggestions[col] = 'numeric'
            continue
        except:
            pass
        
        # Sprawdź czy kolumna wygląda na datę
        try:
            pd.to_datetime(data[col], errors='raise')
            suggestions[col] = 'datetime'
            continue
        except:
            pass
        
        # Sprawdź czy kolumna wygląda na boolean
        unique_vals = data[col].dropna().astype(str).str.lower().unique()
        if len(unique_vals) <= 2 and all(val in ['true', 'false', 't', 'f', 'yes', 'no', 'y', 'n', '1', '0', '+', '-'] for val in unique_vals):
            suggestions[col] = 'boolean'
            continue
        
        # Sprawdź czy kolumna ma małą liczbę unikalnych wartości (kategorie)
        unique_count = data[col].nunique()
        total_count = len(data[col].dropna())
        if unique_count / total_count < 0.05 or unique_count < 10:
            suggestions[col] = 'categorical'
        else:
            suggestions[col] = 'text'
    
    return suggestions

def suggest_data_preprocessing(data):
    """Sugeruje operacje preprocessing na podstawie analizy danych."""
    if data is None:
        return []
    
    suggestions = []
    
    # Sprawdź brakujące wartości
    missing_data = data.isnull().sum()
    total_missing = missing_data.sum()
    
    if total_missing > 0:
        high_missing_cols = missing_data[missing_data > len(data) * 0.5].index.tolist()
        if high_missing_cols:
            suggestions.append(f"Rozważ usunięcie kolumn z więcej niż 50% brakujących wartości: {high_missing_cols}")
        
        medium_missing_cols = missing_data[(missing_data > 0) & (missing_data <= len(data) * 0.5)].index.tolist()
        if medium_missing_cols:
            suggestions.append(f"Rozważ wypełnienie brakujących wartości w kolumnach: {medium_missing_cols}")
    
    # Sprawdź duplikaty
    duplicates = data.duplicated().sum()
    if duplicates > 0:
        suggestions.append(f"Znaleziono {duplicates} zduplikowanych wierszy - rozważ ich usunięcie")
    
    # Sprawdź kolumny z jedną wartością
    constant_cols = []
    for col in data.columns:
        if data[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        suggestions.append(f"Kolumny z jedną wartością (można usunąć): {constant_cols}")
    
    # Sprawdź kolumny numeryczne do skalowania
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) > 1:
        # Sprawdź różnice w skalach
        ranges = {}
        for col in numeric_cols:
            col_range = data[col].max() - data[col].min()
            if not pd.isna(col_range):
                ranges[col] = col_range
        
        if ranges:
            max_range = max(ranges.values())
            min_range = min(ranges.values())
            if max_range / min_range > 100:  # Różnica większa niż 100x
                suggestions.append("Kolumny numeryczne mają bardzo różne skale - rozważ skalowanie danych")
    
    # Sprawdź kolumny kategoryczne do kodowania
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    high_cardinality_cols = []
    for col in categorical_cols:
        unique_count = data[col].nunique()
        if unique_count > 10:
            high_cardinality_cols.append(f"{col} ({unique_count} wartości)")
    
    if high_cardinality_cols:
        suggestions.append(f"Kolumny kategoryczne z wysoką kardynalnością: {high_cardinality_cols}")
    
    return suggestions

def validate_csv_structure(data):
    """Waliduje strukturę wczytanego CSV."""
    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    if data is None or data.empty:
        validation['is_valid'] = False
        validation['errors'].append("Plik jest pusty")
        return validation
    
    # Sprawdź rozmiar
    if data.shape[0] < 2:
        validation['warnings'].append("Bardzo mała liczba wierszy (mniej niż 2)")
    
    if data.shape[1] < 2:
        validation['warnings'].append("Bardzo mała liczba kolumn (mniej niż 2)")
    
    # Sprawdź nazwy kolumn
    duplicate_cols = data.columns[data.columns.duplicated()].tolist()
    if duplicate_cols:
        validation['errors'].append(f"Zduplikowane nazwy kolumn: {duplicate_cols}")
        validation['is_valid'] = False
    
    # Sprawdź procent brakujących wartości
    missing_percent = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
    if missing_percent > 80:
        validation['errors'].append(f"Zbyt dużo brakujących wartości: {missing_percent:.1f}%")
        validation['is_valid'] = False
    elif missing_percent > 50:
        validation['warnings'].append(f"Dużo brakujących wartości: {missing_percent:.1f}%")
    elif missing_percent > 0:
        validation['info'].append(f"Brakujące wartości: {missing_percent:.1f}%")
    
    # Sprawdź typy danych
    type_info = detect_column_types(data)
    validation['info'].append(f"Wykryte typy kolumn: {type_info}")
    
    return validation

def get_column_mapping():
    """Zwraca mapowanie oryginalnych nazw kolumn na proponowane nazwy."""
    return {
        'A1': 'Płeć',
        'A2': 'Wiek',
        'A3': 'Stosunek_zadłużenia',
        'A4': 'Status_cywilny',
        'A5': 'Typ_klienta_banku',
        'A6': 'Poziom_wykształcenia',
        'A7': 'Branża_zatrudnienia',
        'A8': 'Staż_zatrudnienia',
        'A9': 'Ma_rachunek_RO',
        'A10': 'Ma_rachunek_OS',
        'A11': 'Liczba_aktywnych_kredytów',
        'A12': 'Posiada_inne_zobowiązania',
        'A13': 'Cel_kredytu',
        'A14': 'Długość_historii_kredytowej',
        'A15': 'Roczny_dochód',
        'A16': 'Decyzja_przyznania_kredytu'
    }
