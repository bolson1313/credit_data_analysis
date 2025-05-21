import pandas as pd
import numpy as np
import streamlit as st

def load_csv(file):
    """Wczytuje dane z pliku CSV."""
    try:
        data = pd.read_csv(file, na_values='?')
        
        # Sprawdź czy kolumny są już w formacie A1, A2, itd.
        expected_columns = [f'A{i}' for i in range(1, 17)]
        
        if data.shape[1] == 16 and not all(col in data.columns for col in expected_columns):
            # Jeśli mamy 16 kolumn, ale nie są nazwane A1-A16, zmień ich nazwy
            data.columns = [f'A{i}' for i in range(1, 17)]
            
        return data
    except Exception as e:
        st.error(f"Błąd wczytywania pliku: {e}")
        return None

def load_sample_data():
    """Wczytuje przykładowy zbiór Credit Approval."""
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
        data = pd.read_csv(url, header=None, na_values='?')
        # Ustawienie nazw kolumn zgodnie z opisem zbioru danych
        data.columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8',
                        'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']
        return data
    except Exception as e:
        st.error(f"Błąd wczytywania przykładowych danych: {e}")
        return None

def get_dataset_info(data):
    """Zwraca podstawowe informacje o zbiorze danych."""
    if data is None:
        return None

    info = {
        "rows": data.shape[0],
        "columns": data.shape[1],
        "columns_names": data.columns.tolist(),
        "missing_values": data.isna().sum().sum(),
        "duplicated_rows": data.duplicated().sum(),
        "dtypes": data.dtypes
    }

    return info

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

def get_column_descriptions():
    """Zwraca opisy kolumn."""
    return {
        'A1': 'b / a – dwie kategorie płci',
        'A2': 'wartość ciągła (lata)',
        'A3': 'wartość ciągła (np. dług/dochód)',
        'A4': 'u, y, l, t – status małżeński',
        'A5': 'g, p, gg – kategoria klienta',
        'A6': 'wiele poziomów (np. c, d, cc … ff)',
        'A7': 'kategoria branży/zawodu',
        'A8': 'wartość ciągła (lata pracy)',
        'A9': 't / f – czy posiada rachunek rozliczeniowy',
        'A10': 't / f – czy posiada rachunek oszczędnościowy',
        'A11': 'wartość całkowita (np. liczba innych kredytów)',
        'A12': 't / f – czy ma inne zobowiązania',
        'A13': 'g, p, s – kategoria celu (np. g = ogólny, p = pojazd, s = edukacja)',
        'A14': 'wartość całkowita (miesiące)',
        'A15': 'wartość całkowita (waluta)',
        'A16': '+ / - – zaakceptowany / odrzucony'
    }

def get_original_column_name(display_name):
    """Zwraca oryginalną nazwę kolumny na podstawie nazwy wyświetlanej"""
    column_mapping = get_column_mapping()
    reverse_mapping = {v: k for k, v in column_mapping.items()}
    return reverse_mapping.get(display_name, display_name)

def get_display_column_name(original_name):
    """Zwraca nazwę wyświetlaną na podstawie oryginalnej nazwy kolumny"""
    column_mapping = get_column_mapping()
    return column_mapping.get(original_name, original_name)