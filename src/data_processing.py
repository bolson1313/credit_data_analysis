import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from category_encoders import BinaryEncoder


def show_message(message, msg_type="info"):
    """Wyświetla wiadomość w Streamlit."""
    if msg_type == "success":
        st.success(message)
    elif msg_type == "error":
        st.error(message)
    elif msg_type == "warning":
        st.warning(message)
    else:
        st.info(message)


def remove_rows_by_indices(data, indices_str):
    """
    Usuwa wiersze według indeksów.
    
    Args:
        data: DataFrame
        indices_str: String z indeksami "1,3,5" lub "1-5" lub "1,3-5,7"
    
    Returns:
        DataFrame: Dane bez usuniętych wierszy
    """
    if data is None or data.empty:
        show_message("Brak danych do przetworzenia", "error")
        return data
    
    if not indices_str or indices_str.strip() == "":
        show_message("Nie podano indeksów do usunięcia", "warning")
        return data
    
    try:
        # Parsowanie indeksów
        indices_to_remove = set()
        parts = [p.strip() for p in indices_str.split(',')]
        
        for part in parts:
            if '-' in part:
                # Zakres np. "1-5"
                start, end = map(int, part.split('-'))
                indices_to_remove.update(range(start, end + 1))
            else:
                # Pojedynczy indeks
                indices_to_remove.add(int(part))
        
        # Filtruj tylko prawidłowe indeksy
        valid_indices = [i for i in indices_to_remove if 0 <= i < len(data)]
        
        if not valid_indices:
            show_message("Brak prawidłowych indeksów do usunięcia", "warning")
            return data
        
        # Usuń wiersze
        result = data.drop(data.index[valid_indices]).reset_index(drop=True)
        
        show_message(f"✅ Usunięto {len(valid_indices)} wierszy. Zostało {len(result)} wierszy.", "success")
        return result
        
    except Exception as e:
        show_message(f"❌ Błąd podczas usuwania wierszy: {str(e)}", "error")
        return data


def keep_rows_by_indices(data, indices_str):
    """
    Zachowuje tylko wybrane wiersze według indeksów.
    """
    if data is None or data.empty:
        show_message("Brak danych do przetworzenia", "error")
        return data
    
    if not indices_str or indices_str.strip() == "":
        show_message("Nie podano indeksów do zachowania", "warning")
        return data
    
    try:
        # Parsowanie indeksów
        indices_to_keep = set()
        parts = [p.strip() for p in indices_str.split(',')]
        
        for part in parts:
            if '-' in part:
                start, end = map(int, part.split('-'))
                indices_to_keep.update(range(start, end + 1))
            else:
                indices_to_keep.add(int(part))
        
        # Filtruj tylko prawidłowe indeksy
        valid_indices = [i for i in indices_to_keep if 0 <= i < len(data)]
        
        if not valid_indices:
            show_message("Brak prawidłowych indeksów do zachowania", "warning")
            return data
        
        # Zachowaj tylko wybrane wiersze
        result = data.iloc[valid_indices].reset_index(drop=True)
        
        show_message(f"✅ Zachowano {len(valid_indices)} wierszy.", "success")
        return result
        
    except Exception as e:
        show_message(f"❌ Błąd podczas wybierania wierszy: {str(e)}", "error")
        return data


def remove_rows_by_value(data, column, value):
    """Usuwa wiersze gdzie kolumna ma określoną wartość."""
    if data is None or data.empty:
        show_message("Brak danych do przetworzenia", "error")
        return data
    
    if column not in data.columns:
        show_message(f"Kolumna '{column}' nie istnieje", "error")
        return data
    
    try:
        # Znajdź wiersze do usunięcia
        mask = data[column].astype(str) == str(value)
        rows_to_remove = mask.sum()
        
        if rows_to_remove == 0:
            show_message(f"Nie znaleziono wierszy z wartością '{value}' w kolumnie '{column}'", "warning")
            return data
        
        # Usuń wiersze
        result = data[~mask].reset_index(drop=True)
        
        show_message(f"✅ Usunięto {rows_to_remove} wierszy z wartością '{value}'. Zostało {len(result)} wierszy.", "success")
        return result
        
    except Exception as e:
        show_message(f"❌ Błąd podczas usuwania wierszy: {str(e)}", "error")
        return data


def keep_rows_by_value(data, column, value):
    """Zachowuje tylko wiersze gdzie kolumna ma określoną wartość."""
    if data is None or data.empty:
        show_message("Brak danych do przetworzenia", "error")
        return data
    
    if column not in data.columns:
        show_message(f"Kolumna '{column}' nie istnieje", "error")
        return data
    
    try:
        # Znajdź wiersze do zachowania
        mask = data[column].astype(str) == str(value)
        rows_to_keep = mask.sum()
        
        if rows_to_keep == 0:
            show_message(f"Nie znaleziono wierszy z wartością '{value}' w kolumnie '{column}'", "warning")
            return data
        
        # Zachowaj tylko wybrane wiersze
        result = data[mask].reset_index(drop=True)
        
        show_message(f"✅ Zachowano {rows_to_keep} wierszy z wartością '{value}'.", "success")
        return result
        
    except Exception as e:
        show_message(f"❌ Błąd podczas wybierania wierszy: {str(e)}", "error")
        return data


def remove_columns(data, columns_to_remove):
    """Usuwa wybrane kolumny."""
    if data is None or data.empty:
        show_message("Brak danych do przetworzenia", "error")
        return data
    
    if not columns_to_remove:
        show_message("Nie wybrano kolumn do usunięcia", "warning")
        return data
    
    try:
        # Sprawdź które kolumny istnieją
        existing_cols = [col for col in columns_to_remove if col in data.columns]
        missing_cols = [col for col in columns_to_remove if col not in data.columns]
        
        if missing_cols:
            show_message(f"Nie znaleziono kolumn: {missing_cols}", "warning")
        
        if not existing_cols:
            show_message("Brak prawidłowych kolumn do usunięcia", "warning")
            return data
        
        # Usuń kolumny
        result = data.drop(columns=existing_cols)
        
        show_message(f"✅ Usunięto kolumny: {existing_cols}", "success")
        return result
        
    except Exception as e:
        show_message(f"❌ Błąd podczas usuwania kolumn: {str(e)}", "error")
        return data


def replace_values(data, column, old_value, new_value):
    """Zamienia wartości w kolumnie."""
    if data is None or data.empty:
        show_message("Brak danych do przetworzenia", "error")
        return data
    
    if column not in data.columns:
        show_message(f"Kolumna '{column}' nie istnieje", "error")
        return data
    
    try:
        result = data.copy()
        
        # Znajdź wartości do zamiany
        mask = result[column].astype(str) == str(old_value)
        count = mask.sum()
        
        if count == 0:
            show_message(f"Nie znaleziono wartości '{old_value}' w kolumnie '{column}'", "warning")
            return data
        
        # Zamień wartości
        result.loc[mask, column] = new_value
        
        show_message(f"✅ Zamieniono {count} wystąpień '{old_value}' na '{new_value}' w kolumnie '{column}'", "success")
        return result
        
    except Exception as e:
        show_message(f"❌ Błąd podczas zamiany wartości: {str(e)}", "error")
        return data


def handle_missing_values(data, method='drop_rows', columns=None):
    """Obsługuje brakujące wartości."""
    if data is None or data.empty:
        show_message("Brak danych do przetworzenia", "error")
        return data
    
    try:
        result = data.copy()
        
        # Wybór kolumn
        if columns is None:
            target_columns = data.columns.tolist()
        else:
            target_columns = [col for col in columns if col in data.columns]
        
        if not target_columns:
            show_message("Brak prawidłowych kolumn", "warning")
            return data
        
        # Sprawdź brakujące wartości
        missing_before = result[target_columns].isna().sum().sum()
        
        if missing_before == 0:
            show_message("Brak brakujących wartości w wybranych kolumnach", "info")
            return data
        
        if method == 'drop_rows':
            # Usuń wiersze z brakującymi wartościami
            initial_rows = len(result)
            result = result.dropna(subset=target_columns).reset_index(drop=True)
            removed_rows = initial_rows - len(result)
            show_message(f"✅ Usunięto {removed_rows} wierszy z brakującymi wartościami", "success")
            
        elif method == 'drop_columns':
            # Usuń kolumny z brakującymi wartościami
            cols_with_missing = [col for col in target_columns if result[col].isna().any()]
            if cols_with_missing:
                result = result.drop(columns=cols_with_missing)
                show_message(f"✅ Usunięto kolumny: {cols_with_missing}", "success")
            else:
                show_message("Brak kolumn z brakującymi wartościami", "info")
                
        elif method in ['mean', 'median', 'mode', 'zero']:
            # Wypełnij brakujące wartości
            filled_cols = []
            for col in target_columns:
                if result[col].isna().any():
                    if pd.api.types.is_numeric_dtype(result[col]):
                        if method == 'mean':
                            fill_value = result[col].mean()
                        elif method == 'median':
                            fill_value = result[col].median()
                        elif method == 'zero':
                            fill_value = 0
                        else:  # mode
                            mode_val = result[col].mode()
                            fill_value = mode_val[0] if not mode_val.empty else 0
                    else:
                        # Dla kolumn tekstowych
                        if method == 'zero':
                            fill_value = ""
                        else:
                            mode_val = result[col].mode()
                            fill_value = mode_val[0] if not mode_val.empty else "Unknown"
                    
                    result[col] = result[col].fillna(fill_value)
                    filled_cols.append(col)
            
            if filled_cols:
                show_message(f"✅ Wypełniono brakujące wartości w kolumnach: {filled_cols}", "success")
        
        return result
        
    except Exception as e:
        show_message(f"❌ Błąd podczas obsługi brakujących wartości: {str(e)}", "error")
        return data


def remove_duplicates(data):
    """Usuwa duplikaty."""
    if data is None or data.empty:
        show_message("Brak danych do przetworzenia", "error")
        return data
    
    try:
        initial_count = len(data)
        result = data.drop_duplicates().reset_index(drop=True)
        removed_count = initial_count - len(result)
        
        if removed_count > 0:
            show_message(f"✅ Usunięto {removed_count} duplikatów", "success")
        else:
            show_message("Nie znaleziono duplikatów", "info")
        
        return result
        
    except Exception as e:
        show_message(f"❌ Błąd podczas usuwania duplikatów: {str(e)}", "error")
        return data


def scale_data(data, columns, method='minmax'):
    """Skaluje dane numeryczne."""
    if data is None or data.empty:
        show_message("Brak danych do przetworzenia", "error")
        return data
    
    if not columns:
        show_message("Nie wybrano kolumn do skalowania", "warning")
        return data
    
    try:
        result = data.copy()
        
        # Sprawdź kolumny numeryczne
        numeric_cols = []
        for col in columns:
            if col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    numeric_cols.append(col)
                else:
                    show_message(f"Kolumna '{col}' nie jest numeryczna", "warning")
            else:
                show_message(f"Kolumna '{col}' nie istnieje", "warning")
        
        if not numeric_cols:
            show_message("Brak numerycznych kolumn do skalowania", "error")
            return data
        
        # Skalowanie
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            show_message(f"Nieznana metoda skalowania: {method}", "error")
            return data
        
        # Usuń wiersze z brakującymi wartościami w kolumnach do skalowania
        mask = result[numeric_cols].notna().all(axis=1)
        if mask.sum() == 0:
            show_message("Brak wierszy bez brakujących wartości", "error")
            return data
        
        # Skaluj dane
        result.loc[mask, numeric_cols] = scaler.fit_transform(result.loc[mask, numeric_cols])
        
        method_name = "Min-Max" if method == 'minmax' else "Standard"
        show_message(f"✅ Przeskalowano kolumny {numeric_cols} metodą {method_name}", "success")
        
        return result
        
    except Exception as e:
        show_message(f"❌ Błąd podczas skalowania: {str(e)}", "error")
        return data


def encode_categorical(data, columns, method='onehot'):
    """Koduje kolumny kategoryczne."""
    if data is None or data.empty:
        show_message("Brak danych do przetworzenia", "error")
        return data
    
    if not columns:
        show_message("Nie wybrano kolumn do kodowania", "warning")
        return data
    
    try:
        result = data.copy()
        
        # Sprawdź kolumny
        valid_cols = [col for col in columns if col in data.columns]
        invalid_cols = [col for col in columns if col not in data.columns]
        
        if invalid_cols:
            show_message(f"Nie znaleziono kolumn: {invalid_cols}", "warning")
        
        if not valid_cols:
            show_message("Brak prawidłowych kolumn do kodowania", "error")
            return data
        
        # Przygotuj dane
        for col in valid_cols:
            result[col] = result[col].astype(str).fillna('Missing')
        
        if method == 'onehot':
            # One-hot encoding
            result = pd.get_dummies(result, columns=valid_cols, drop_first=False)
            show_message(f"✅ Zakodowano kolumny {valid_cols} metodą One-Hot", "success")
            
        elif method == 'binary':
            # Binary encoding
            encoder = BinaryEncoder(cols=valid_cols, return_df=True)
            encoded = encoder.fit_transform(result[valid_cols])
            
            # Usuń oryginalne kolumny i dodaj zakodowane
            result = result.drop(columns=valid_cols)
            result = pd.concat([result, encoded], axis=1)
            show_message(f"✅ Zakodowano kolumny {valid_cols} metodą Binary", "success")
        
        else:
            show_message(f"Nieznana metoda kodowania: {method}", "error")
            return data
        
        return result
        
    except Exception as e:
        show_message(f"❌ Błąd podczas kodowania: {str(e)}", "error")
        return data