import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from category_encoders import BinaryEncoder
from .data_loader import get_display_column_name


def remove_rows(data, indices_str):
    """
    Usuwa wiersze według indeksów. Akceptuje różne formaty:
    - Pojedyncze liczby: "1,3,5"
    - Zakresy: "1-5"
    - Kombinacje: "1,3-5,7,10-12"
    """
    if data is None or not indices_str:
        print("Nie podano indeksów do usunięcia")
        return data

    try:
        # Inicjalizacja listy indeksów
        indices_to_remove = set()
        
        # Podziel string na części po przecinku
        parts = [p.strip() for p in str(indices_str).split(',')]
        
        print(f"\nPrzetwarzanie indeksów: {indices_str}")
        
        for part in parts:
            if '-' in part:
                # Obsługa zakresu (np. "5-10")
                try:
                    start, end = map(int, part.split('-'))
                    if start <= end:
                        indices_to_remove.update(range(start, end + 1))
                        print(f"Dodano zakres {start}-{end}")
                    else:
                        print(f"Nieprawidłowy zakres (start > end): {part}")
                except ValueError:
                    print(f"Pominięto nieprawidłowy zakres: {part}")
                    continue
            else:
                # Obsługa pojedynczej liczby
                try:
                    idx = int(part)
                    indices_to_remove.add(idx)
                    print(f"Dodano indeks: {idx}")
                except ValueError:
                    print(f"Pominięto nieprawidłową wartość: {part}")
                    continue
        
        # Konwersja na listę i sprawdzenie zakresu
        valid_indices = sorted([i for i in indices_to_remove if 0 <= i < len(data)])
        
        if not valid_indices:
            print("Brak prawidłowych indeksów do usunięcia")
            return data
        
        # Zapisanie oryginalnych danych z usuwanych wierszy
        removed_data = data.iloc[valid_indices].copy()
        
        # Tworzenie kopii danych i usuwanie wierszy
        result = data.copy()
        result = result.drop(index=result.index[valid_indices]).reset_index(drop=True)
        
        # Wyświetlenie szczegółowych informacji
        print(f"\nOperacja usuwania:")
        print(f"- Usunięto {len(valid_indices)} wierszy")
        print(f"- Indeksy usuniętych wierszy: {valid_indices}")
        print(f"- Liczba wierszy przed: {len(data)}")
        print(f"- Liczba wierszy po: {len(result)}")
        print("\nUsunięte wiersze:")
        print(removed_data)
        
        return result
        
    except Exception as e:
        print(f"Błąd podczas usuwania wierszy: {str(e)}")
        return data


def remove_rows_by_value(data, column, value):
    """Usuwa wiersze gdzie w kolumnie występuje określona wartość."""
    if data is None or column not in data.columns:
        return data

    try:
        # Tworzenie kopii danych
        result = data.copy()
        # Tworzenie maski dla wierszy do usunięcia
        mask = result[column].astype(str) == str(value)
        # Usuwanie wierszy spełniających warunek
        result = result[~mask].reset_index(drop=True)
        return result
        
    except Exception as e:
        print(f"Błąd podczas usuwania wierszy: {str(e)}")
        return data


def remove_columns(data, columns):
    """Usuwa kolumny według nazw."""
    if data is None or not columns:
        return data

    try:
        # Sprawdzenie, czy kolumny istnieją
        valid_columns = [col for col in columns if col in data.columns]
        if not valid_columns:
            return data

        # Tworzenie kopii danych
        result = data.copy()
        # Usuwanie kolumn
        result = result.drop(columns=valid_columns)
        return result
        
    except Exception as e:
        print(f"Błąd podczas usuwania kolumn: {str(e)}")
        return data


def replace_values(data, column, old_value, new_value):
    """Zamienia wartości w kolumnie."""
    if data is None or column not in data.columns:
        return data

    try:
        # Tworzenie kopii danych
        result = data.copy()
        # Zamiana wartości
        result[column] = result[column].replace(str(old_value), str(new_value))
        return result
        
    except Exception as e:
        print(f"Błąd podczas zamiany wartości: {str(e)}")
        return data


def replace_values_regex(data, column, pattern, new_value):
    """Zamienia wartości pasujące do wzorca regex w kolumnie."""
    if data is None or column not in data.columns:
        return data

    try:
        # Tworzenie kopii danych
        result = data.copy()
        # Zamiana wartości według wzorca
        result[column] = result[column].astype(str).replace(pattern, new_value, regex=True)
        return result
        
    except Exception as e:
        print(f"Błąd podczas zamiany wartości: {str(e)}")
        return data


def handle_missing_values(data, method='drop', columns=None):
    """Obsługuje brakujące wartości w danych."""
    if data is None:
        return data

    try:
        # Tworzenie kopii danych
        result = data.copy()

        # Jeśli nie podano kolumn, użyj wszystkich
        if columns is None:
            columns = data.columns
        else:
            columns = [col for col in columns if col in data.columns]

        if not columns:
            return data

        if method == 'drop_rows':
            # Usuń wiersze z brakującymi wartościami w określonych kolumnach
            result = result.dropna(subset=columns).reset_index(drop=True)
        elif method == 'drop_columns':
            # Usuń kolumny z brakującymi wartościami
            cols_to_drop = [col for col in columns if result[col].isna().any()]
            result = result.drop(columns=cols_to_drop)
        else:
            # Wypełnij brakujące wartości różnymi metodami
            for col in columns:
                if result[col].isna().any():
                    if pd.api.types.is_numeric_dtype(result[col]):
                        if method == 'mean':
                            result[col] = result[col].fillna(result[col].mean())
                        elif method == 'median':
                            result[col] = result[col].fillna(result[col].median())
                        elif method == 'zero':
                            result[col] = result[col].fillna(0)
                        else:  # 'mode'
                            result[col] = result[col].fillna(result[col].mode()[0] if not result[col].mode().empty else 0)
                    else:
                        # Dla kolumn nie-numerycznych użyj mody
                        result[col] = result[col].fillna(result[col].mode()[0] if not result[col].mode().empty else '')

        return result
        
    except Exception as e:
        print(f"Błąd podczas obsługi brakujących wartości: {str(e)}")
        return data


def remove_duplicates(data):
    """Usuwa duplikaty wierszy."""
    if data is None:
        return data

    try:
        return data.drop_duplicates().reset_index(drop=True)
    except Exception as e:
        print(f"Błąd podczas usuwania duplikatów: {str(e)}")
        return data


def scale_data(data, columns, method='minmax'):
    """Skaluje dane numeryczne różnymi metodami."""
    if data is None or not columns:
        return data

    try:
        # Tworzenie kopii danych
        result = data.copy()

        # Sprawdzenie, czy kolumny są numeryczne
        valid_columns = [col for col in columns if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
        
        if not valid_columns:
            return result

        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            return result

        # Skalowanie danych
        result[valid_columns] = scaler.fit_transform(result[valid_columns])
        return result
        
    except Exception as e:
        print(f"Błąd podczas skalowania danych: {str(e)}")
        return data


def encode_categorical(data, columns, method='onehot'):
    """Koduje kolumny kategoryczne."""
    if data is None or not columns:
        return data

    try:
        # Tworzenie kopii danych
        result = data.copy()

        # Sprawdzenie, czy kolumny istnieją
        valid_columns = [col for col in columns if col in data.columns]
        
        if not valid_columns:
            return result

        if method == 'onehot':
            # Kodowanie one-hot
            result = pd.get_dummies(result, columns=valid_columns, drop_first=False)
        elif method == 'binary':
            # Kodowanie binarne
            encoder = BinaryEncoder(cols=valid_columns)
            encoded = encoder.fit_transform(result)
            # Usunięcie oryginalnych kolumn i zastąpienie zakodowanymi
            result = result.drop(columns=valid_columns)
            result = pd.concat([result, encoded], axis=1)

        return result
        
    except Exception as e:
        print(f"Błąd podczas kodowania danych: {str(e)}")
        return data


def select_rows(data, indices_str, mode='keep'):
    """
    Wybiera lub usuwa wiersze według indeksów.
    
    Args:
        data: DataFrame z danymi
        indices_str: String z indeksami (np. "1,3-5,7")
        mode: 'keep' aby zachować wybrane wiersze, 'remove' aby je usunąć
    """
    if data is None or not indices_str:
        print("Nie podano indeksów")
        return data

    try:
        # Inicjalizacja zbioru indeksów
        indices_set = set()
        
        # Podziel string na części po przecinku
        parts = [p.strip() for p in str(indices_str).split(',')]
        
        print(f"\nPrzetwarzanie indeksów: {indices_str}")
        
        for part in parts:
            if '-' in part:
                # Obsługa zakresu (np. "5-10")
                try:
                    start, end = map(int, part.split('-'))
                    if start <= end:
                        indices_set.update(range(start, end + 1))
                        print(f"Dodano zakres {start}-{end}")
                    else:
                        print(f"Nieprawidłowy zakres (start > end): {part}")
                except ValueError:
                    print(f"Pominięto nieprawidłowy zakres: {part}")
                    continue
            else:
                # Obsługa pojedynczej liczby
                try:
                    idx = int(part)
                    indices_set.add(idx)
                    print(f"Dodano indeks: {idx}")
                except ValueError:
                    print(f"Pominięto nieprawidłową wartość: {part}")
                    continue
        
        # Konwersja na listę i sprawdzenie zakresu
        valid_indices = sorted([i for i in indices_set if 0 <= i < len(data)])
        
        if not valid_indices:
            print("Brak prawidłowych indeksów")
            return data
        
        # Tworzenie kopii danych
        result = data.copy()
        
        if mode == 'keep':
            # Zachowaj tylko wybrane wiersze
            result = result.iloc[valid_indices].reset_index(drop=True)
            print(f"\nZachowano {len(valid_indices)} wierszy")
        else:
            # Usuń wybrane wiersze
            result = result.drop(index=result.index[valid_indices]).reset_index(drop=True)
            print(f"\nUsunięto {len(valid_indices)} wierszy")
        
        return result
        
    except Exception as e:
        print(f"Błąd podczas przetwarzania wierszy: {str(e)}")
        return data


def replace_values_in_columns(data, replacements):
    """
    Zamienia wartości w wybranych kolumnach według zadanego słownika.
    
    Args:
        data: DataFrame z danymi
        replacements: Lista krotek (kolumna, stara_wartość, nowa_wartość)
    """
    if data is None or not replacements:
        return data

    try:
        # Tworzenie kopii danych
        result = data.copy()
        
        for column, old_value, new_value in replacements:
            if column in result.columns:
                # Zamiana wartości w kolumnie
                mask = (result[column].astype(str) == str(old_value))
                result.loc[mask, column] = new_value
                print(f"Zamieniono wartości '{old_value}' na '{new_value}' w kolumnie {column}")
            else:
                print(f"Kolumna {column} nie istnieje w danych")
        
        return result
        
    except Exception as e:
        print(f"Błąd podczas zamiany wartości: {str(e)}")
        return data