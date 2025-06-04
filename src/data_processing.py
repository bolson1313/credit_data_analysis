import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from category_encoders import BinaryEncoder


def log_message(message, level="info"):
    """Log messages to Streamlit UI instead of console."""
    if level == "info":
        st.info(message)
    elif level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    else:
        st.write(message)


def remove_rows(data, indices_str):
    """
    Usuwa wiersze według indeksów. Akceptuje różne formaty:
    - Pojedyncze liczby: "1,3,5"
    - Zakresy: "1-5"
    - Kombinacje: "1,3-5,7,10-12"
    """
    if data is None or not indices_str:
        log_message("Nie podano indeksów do usunięcia", "warning")
        return data

    try:
        # Upewnij się, że mamy kopię danych
        result = data.copy()
        
        # Inicjalizacja listy indeksów
        indices_to_remove = set()
        
        # Podziel string na części po przecinku
        parts = [p.strip() for p in str(indices_str).split(',')]
        
        log_message(f"Przetwarzanie indeksów: {indices_str}")
        
        for part in parts:
            if '-' in part:
                # Obsługa zakresu (np. "5-10")
                try:
                    start, end = map(int, part.split('-'))
                    if start <= end:
                        indices_to_remove.update(range(start, end + 1))
                        log_message(f"Dodano zakres {start}-{end}")
                    else:
                        log_message(f"Nieprawidłowy zakres (start > end): {part}", "warning")
                except ValueError:
                    log_message(f"Pominięto nieprawidłowy zakres: {part}", "warning")
                    continue
            else:
                # Obsługa pojedynczej liczby
                try:
                    idx = int(part)
                    indices_to_remove.add(idx)
                    log_message(f"Dodano indeks: {idx}")
                except ValueError:
                    log_message(f"Pominięto nieprawidłową wartość: {part}", "warning")
                    continue
        
        # Konwersja na listę i sprawdzenie zakresu
        valid_indices = sorted([i for i in indices_to_remove if 0 <= i < len(result)])
        
        if not valid_indices:
            log_message("Brak prawidłowych indeksów do usunięcia", "warning")
            return result
        
        # Zapisanie oryginalnych danych z usuwanych wierszy dla informacji
        removed_data = result.iloc[valid_indices].copy()
        
        # Usuwanie wierszy
        result = result.drop(index=result.index[valid_indices]).reset_index(drop=True)
        
        # Wyświetlenie szczegółowych informacji
        log_message(f"Usunięto {len(valid_indices)} wierszy. Pozostało {len(result)} wierszy.", "success")
        
        # Opcjonalnie pokaż usunięte wiersze
        if len(valid_indices) <= 10:  # Tylko dla małej liczby usuniętych wierszy
            with st.expander(f"Pokaż usunięte wiersze ({len(valid_indices)})"):
                st.dataframe(removed_data)
        
        return result
        
    except Exception as e:
        log_message(f"Błąd podczas usuwania wierszy: {str(e)}", "error")
        return data


def remove_rows_by_value(data, column, value):
    """Usuwa wiersze gdzie w kolumnie występuje określona wartość."""
    if data is None or column not in data.columns:
        log_message(f"Kolumna {column} nie istnieje w danych", "error")
        return data

    try:
        # Tworzenie kopii danych
        result = data.copy()
        initial_count = len(result)
        
        # Tworzenie maski dla wierszy do usunięcia
        mask = result[column].astype(str) == str(value)
        removed_count = mask.sum()
        
        # Usuwanie wierszy spełniających warunek
        result = result[~mask].reset_index(drop=True)
        
        log_message(f"Usunięto {removed_count} wierszy z wartością '{value}' w kolumnie {column}", "success")
        return result
        
    except Exception as e:
        log_message(f"Błąd podczas usuwania wierszy: {str(e)}", "error")
        return data


def remove_columns(data, columns):
    """Usuwa kolumny według nazw."""
    if data is None or not columns:
        log_message("Nie podano kolumn do usunięcia", "warning")
        return data

    try:
        # Sprawdzenie, czy kolumny istnieją
        valid_columns = [col for col in columns if col in data.columns]
        invalid_columns = [col for col in columns if col not in data.columns]
        
        if invalid_columns:
            log_message(f"Nie znaleziono kolumn: {invalid_columns}", "warning")
        
        if not valid_columns:
            log_message("Brak prawidłowych kolumn do usunięcia", "warning")
            return data

        # Tworzenie kopii danych i usuwanie kolumn
        result = data.copy()
        result = result.drop(columns=valid_columns)
        
        log_message(f"Usunięto kolumny: {valid_columns}", "success")
        return result
        
    except Exception as e:
        log_message(f"Błąd podczas usuwania kolumn: {str(e)}", "error")
        return data


def replace_values(data, column, old_value, new_value):
    """Zamienia wartości w kolumnie."""
    if data is None or column not in data.columns:
        log_message(f"Kolumna {column} nie istnieje w danych", "error")
        return data

    try:
        # Tworzenie kopii danych
        result = data.copy()
        
        # Policz ile wartości zostanie zamienionych
        old_value_str = str(old_value) if old_value is not None else ''
        mask = result[column].astype(str) == old_value_str
        count = mask.sum()
        
        # Zamiana wartości
        result[column] = result[column].replace(old_value, new_value)
        
        log_message(f"Zamieniono {count} wystąpień '{old_value}' na '{new_value}' w kolumnie {column}", "success")
        return result
        
    except Exception as e:
        log_message(f"Błąd podczas zamiany wartości: {str(e)}", "error")
        return data


def replace_values_regex(data, column, pattern, new_value):
    """Zamienia wartości pasujące do wzorca regex w kolumnie."""
    if data is None or column not in data.columns:
        log_message(f"Kolumna {column} nie istnieje w danych", "error")
        return data

    try:
        # Tworzenie kopii danych
        result = data.copy()
        
        # Zamiana wartości według wzorca
        old_data = result[column].copy()
        result[column] = result[column].astype(str).replace(pattern, new_value, regex=True)
        
        # Policz zmiany
        changes = (old_data.astype(str) != result[column].astype(str)).sum()
        
        log_message(f"Zamieniono {changes} wartości według wzorca '{pattern}' na '{new_value}' w kolumnie {column}", "success")
        return result
        
    except Exception as e:
        log_message(f"Błąd podczas zamiany wartości (regex): {str(e)}", "error")
        return data


def handle_missing_values(data, method='drop_rows', columns=None):
    """Obsługuje brakujące wartości w danych."""
    if data is None:
        log_message("Brak danych do przetworzenia", "error")
        return data

    try:
        # Tworzenie kopii danych
        result = data.copy()
        initial_shape = result.shape

        # Jeśli nie podano kolumn, użyj wszystkich
        if columns is None:
            columns = data.columns.tolist()
        else:
            columns = [col for col in columns if col in data.columns]

        if not columns:
            log_message("Brak prawidłowych kolumn do przetworzenia", "warning")
            return data

        # Sprawdź czy są brakujące wartości w wybranych kolumnach
        missing_before = result[columns].isna().sum().sum()
        
        if missing_before == 0:
            log_message("Brak brakujących wartości w wybranych kolumnach", "info")
            return result

        if method == 'drop_rows':
            # Usuń wiersze z brakującymi wartościami w określonych kolumnach
            result = result.dropna(subset=columns).reset_index(drop=True)
            rows_removed = initial_shape[0] - result.shape[0]
            log_message(f"Usunięto {rows_removed} wierszy z brakującymi wartościami", "success")
            
        elif method == 'drop_columns':
            # Usuń kolumny z brakującymi wartościami
            cols_to_drop = [col for col in columns if result[col].isna().any()]
            if cols_to_drop:
                result = result.drop(columns=cols_to_drop)
                log_message(f"Usunięto kolumny: {cols_to_drop}", "success")
            else:
                log_message("Brak kolumn z brakującymi wartościami do usunięcia", "info")
                
        else:
            # Wypełnij brakujące wartości różnymi metodami
            filled_info = []
            for col in columns:
                if result[col].isna().any():
                    missing_count = result[col].isna().sum()
                    
                    if pd.api.types.is_numeric_dtype(result[col]):
                        if method == 'mean':
                            fill_value = result[col].mean()
                            result[col] = result[col].fillna(fill_value)
                            filled_info.append(f"{col}: {missing_count} wartości wypełniono średnią ({fill_value:.2f})")
                        elif method == 'median':
                            fill_value = result[col].median()
                            result[col] = result[col].fillna(fill_value)
                            filled_info.append(f"{col}: {missing_count} wartości wypełniono medianą ({fill_value:.2f})")
                        elif method == 'zero':
                            result[col] = result[col].fillna(0)
                            filled_info.append(f"{col}: {missing_count} wartości wypełniono zerem")
                        else:  # 'mode'
                            mode_val = result[col].mode()
                            fill_value = mode_val[0] if not mode_val.empty else 0
                            result[col] = result[col].fillna(fill_value)
                            filled_info.append(f"{col}: {missing_count} wartości wypełniono modą ({fill_value})")
                    else:
                        # Dla kolumn nie-numerycznych użyj mody
                        mode_val = result[col].mode()
                        fill_value = mode_val[0] if not mode_val.empty else 'Unknown'
                        result[col] = result[col].fillna(fill_value)
                        filled_info.append(f"{col}: {missing_count} wartości wypełniono modą ('{fill_value}')")
            
            if filled_info:
                log_message("Wypełniono brakujące wartości:", "success")
                for info in filled_info:
                    st.write(f"• {info}")

        # Sprawdź efekt
        missing_after = result.isna().sum().sum()
        log_message(f"Brakujące wartości: {missing_before} → {missing_after}", "info")
        
        return result
        
    except Exception as e:
        log_message(f"Błąd podczas obsługi brakujących wartości: {str(e)}", "error")
        return data


def remove_duplicates(data):
    """Usuwa duplikaty wierszy."""
    if data is None:
        log_message("Brak danych do przetworzenia", "error")
        return data

    try:
        initial_count = len(data)
        result = data.drop_duplicates().reset_index(drop=True)
        removed_count = initial_count - len(result)
        
        if removed_count > 0:
            log_message(f"Usunięto {removed_count} zduplikowanych wierszy", "success")
        else:
            log_message("Nie znaleziono duplikatów", "info")
            
        return result
    except Exception as e:
        log_message(f"Błąd podczas usuwania duplikatów: {str(e)}", "error")
        return data


def scale_data(data, columns, method='minmax'):
    """Skaluje dane numeryczne różnymi metodami."""
    if data is None or not columns:
        log_message("Brak danych lub kolumn do skalowania", "warning")
        return data

    try:
        # Tworzenie kopii danych
        result = data.copy()

        # Sprawdzenie, czy kolumny istnieją i są numeryczne
        valid_columns = []
        for col in columns:
            if col not in data.columns:
                log_message(f"Kolumna {col} nie istnieje", "warning")
                continue
            
            # Spróbuj przekonwertować na numeryczne
            try:
                result[col] = pd.to_numeric(result[col], errors='coerce')
                if result[col].notna().sum() > 0:  # Czy są jakieś prawidłowe wartości numeryczne
                    valid_columns.append(col)
                else:
                    log_message(f"Kolumna {col} nie zawiera prawidłowych wartości numerycznych", "warning")
            except:
                log_message(f"Nie można przekonwertować kolumny {col} na numeryczną", "warning")
        
        if not valid_columns:
            log_message("Brak prawidłowych kolumn numerycznych do skalowania", "error")
            return result

        # Sprawdź czy są brakujące wartości w kolumnach do skalowania
        missing_values = result[valid_columns].isna().sum().sum()
        if missing_values > 0:
            log_message(f"Uwaga: {missing_values} brakujących wartości w kolumnach do skalowania. Zostaną pominięte.", "warning")

        if method == 'minmax':
            scaler = MinMaxScaler()
            scaler_name = "Min-Max"
        elif method == 'standard':
            scaler = StandardScaler()
            scaler_name = "Standard (Z-score)"
        else:
            log_message(f"Nieznana metoda skalowania: {method}", "error")
            return result

        # Skalowanie danych (tylko dla wierszy bez brakujących wartości)
        mask = result[valid_columns].notna().all(axis=1)
        if mask.sum() == 0:
            log_message("Brak wierszy bez brakujących wartości do skalowania", "error")
            return result
            
        result.loc[mask, valid_columns] = scaler.fit_transform(result.loc[mask, valid_columns])
        
        log_message(f"Przeskalowano kolumny {valid_columns} metodą {scaler_name}", "success")
        log_message(f"Przeskalowano {mask.sum()} wierszy (pominięto {(~mask).sum()} z brakującymi wartościami)", "info")
        
        return result
        
    except Exception as e:
        log_message(f"Błąd podczas skalowania danych: {str(e)}", "error")
        return data


def encode_categorical(data, columns, method='onehot'):
    """Koduje kolumny kategoryczne metodą onehot lub binary."""
    if data is None or not columns:
        log_message("Brak danych lub kolumn do zakodowania", "warning")
        return data

    try:
        result = data.copy()
        valid_columns = [col for col in columns if col in result.columns]
        invalid_columns = [col for col in columns if col not in result.columns]

        if invalid_columns:
            log_message(f"Nie znaleziono kolumn: {invalid_columns}", "warning")

        if not valid_columns:
            log_message("Brak poprawnych kolumn do zakodowania", "error")
            return result

        # Sprawdź i przygotuj kolumny do kodowania
        prepared_columns = []
        for col in valid_columns:
            # Konwertuj na string i wypełnij brakujące wartości
            result[col] = result[col].astype(str).replace('nan', 'Missing')
            unique_values = result[col].nunique()
            if unique_values > 50:
                log_message(f"Uwaga: Kolumna {col} ma {unique_values} unikalnych wartości. To może utworzyć wiele nowych kolumn.", "warning")
            prepared_columns.append(col)

        if method == 'onehot':
            # One-hot encoding
            original_cols = len(result.columns)
            result = pd.get_dummies(result, columns=prepared_columns, drop_first=False, dtype=float)
            new_cols = len(result.columns)
            log_message(f"One-hot encoding: {len(prepared_columns)} kolumn → {new_cols - original_cols + len(prepared_columns)} nowych kolumn", "success")

        elif method == 'binary':
            # Binary encoding
            encoder = BinaryEncoder(cols=prepared_columns, return_df=True)
            encoded = encoder.fit_transform(result[prepared_columns])

            # Usuń oryginalne kolumny
            result = result.drop(columns=prepared_columns)
            encoded = encoded.astype(float)

            # Połącz zakodowane dane z resztą
            result = pd.concat([result.reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)
            log_message(f"Binary encoding: {len(prepared_columns)} kolumn → {len(encoded.columns)} nowych kolumn", "success")

        else:
            log_message(f"Nieobsługiwana metoda kodowania: {method}", "error")
            return data

        return result

    except Exception as e:
        log_message(f"Błąd podczas kodowania danych ({method}): {str(e)}", "error")
        return data


def select_rows(data, indices_str, mode='keep'):
    """Wybiera lub usuwa wiersze według indeksów."""
    if data is None or not indices_str:
        log_message("Nie podano indeksów", "warning")
        return data

    try:
        # Inicjalizacja zbioru indeksów
        indices_set = set()
        
        # Podziel string na części po przecinku
        parts = [p.strip() for p in str(indices_str).split(',')]
        
        log_message(f"Przetwarzanie indeksów: {indices_str}")
        
        for part in parts:
            if '-' in part:
                try:
                    start, end = map(int, part.split('-'))
                    if start <= end:
                        indices_set.update(range(start, end + 1))
                        log_message(f"Dodano zakres {start}-{end} (włącznie)")
                    else:
                        log_message(f"Nieprawidłowy zakres (start > end): {part}", "warning")
                except ValueError:
                    log_message(f"Pominięto nieprawidłowy zakres: {part}", "warning")
                    continue
            else:
                try:
                    idx = int(part)
                    indices_set.add(idx)
                    log_message(f"Dodano indeks: {idx}")
                except ValueError:
                    log_message(f"Pominięto nieprawidłową wartość: {part}", "warning")
                    continue
        
        # Konwersja na listę i sprawdzenie zakresu
        valid_indices = sorted([i for i in indices_set if 0 <= i < len(data)])
        invalid_indices = [i for i in indices_set if i < 0 or i >= len(data)]
        
        if invalid_indices:
            log_message(f"Pominięto nieprawidłowe indeksy (poza zakresem): {invalid_indices}", "warning")
        
        if not valid_indices:
            log_message("Brak prawidłowych indeksów", "warning")
            return data
        
        # Tworzenie kopii danych
        if mode == 'keep':
            result = data.iloc[valid_indices].copy()
            operation = "zachowano"
        else:
            result = data.drop(data.index[valid_indices]).copy()
            operation = "usunięto"
        
        log_message(f"Operacja zakończona: {operation} {len(valid_indices)} wierszy. Pozostało {len(result)} wierszy.", "success")
        
        return result
        
    except Exception as e:
        log_message(f"Błąd podczas przetwarzania wierszy: {str(e)}", "error")
        return data


def replace_values_in_columns(data, replacements):
    """
    Zamienia wartości w wybranych kolumnach według zadanej listy.
    
    Args:
        data: DataFrame z danymi
        replacements: Lista krotek (kolumna, stara_wartość, nowa_wartość)
    """
    if data is None or not replacements:
        log_message("Brak danych lub zamian do wykonania", "warning")
        return data

    try:
        # Tworzenie kopii danych
        result = data.copy()
        successful_replacements = []
        
        for column, old_value, new_value in replacements:
            if column in result.columns:
                # Policz ile wartości zostanie zamienionych
                old_value_str = str(old_value) if old_value is not None else ''
                mask = (result[column].astype(str) == old_value_str)
                count = mask.sum()
                
                if count > 0:
                    # Zamiana wartości w kolumnie
                    result.loc[mask, column] = new_value
                    successful_replacements.append(f"Kolumna '{column}': {count}× '{old_value}' → '{new_value}'")
                else:
                    log_message(f"Nie znaleziono wartości '{old_value}' w kolumnie '{column}'", "warning")
            else:
                log_message(f"Kolumna '{column}' nie istnieje w danych", "warning")
        
        if successful_replacements:
            log_message("Pomyślnie wykonano zamiany:", "success")
            for replacement in successful_replacements:
                st.write(f"• {replacement}")
        else:
            log_message("Nie wykonano żadnych zamian", "info")
        
        return result
        
    except Exception as e:
        log_message(f"Błąd podczas zamiany wartości: {str(e)}", "error")
        return data


def log_data_changes(original_data, modified_data, operation_type, details=None):
    """Loguje zmiany w danych do UI Streamlit."""
    try:
        changes = {
            'operation_type': operation_type,
            'original_rows': len(original_data),
            'modified_rows': len(modified_data),
            'difference': len(original_data) - len(modified_data),
            'timestamp': pd.Timestamp.now(),
            'details': details or {}
        }
        
        # Wyświetl informacje o zmianach
        if changes['difference'] != 0:
            if changes['difference'] > 0:
                log_message(f"Usunięto {changes['difference']} wierszy", "info")
            else:
                log_message(f"Dodano {abs(changes['difference'])} wierszy", "info")
        
        # Jeśli są szczegóły operacji
        if details:
            with st.expander("Szczegóły operacji"):
                for key, value in details.items():
                    st.write(f"**{key}**: {value}")
        
        return changes
        
    except Exception as e:
        log_message(f"Błąd podczas logowania zmian: {str(e)}", "error")
        return None