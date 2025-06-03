import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Import modułów aplikacji
from modules.data_loader import (
    load_csv, load_sample_data, get_dataset_info,
    get_column_mapping, get_column_descriptions,
    get_original_column_name, get_display_column_name
)
from modules.statistics import calculate_numerical_stats, calculate_categorical_stats, calculate_correlations
from modules.data_processing import (
    remove_rows, remove_columns, replace_values, replace_values_regex,
    handle_missing_values, remove_duplicates, scale_data, encode_categorical,
    select_rows, replace_values_in_columns
)
from modules.visualization import (
    histogram, box_plot, scatter_plot, bar_chart, pie_chart, pair_plot
)
from modules.modeling import (
    prepare_data_for_classification,
    train_classification_model, 
    evaluate_classification_model,
    prepare_data_for_clustering,
    train_kmeans_model,
    evaluate_clustering,
    plot_clusters_2d
)

def get_column_display_names(data):
    """Zwraca słownik mapujący oryginalne nazwy kolumn na nazwy wyświetlane"""
    column_mapping = get_column_mapping()
    
    # Automatyczne mapowanie nazw kolumn, jeśli mamy 16 kolumn i nie są to standardowe nazwy
    if data.shape[1] == 16 and not all(col in column_mapping for col in data.columns):
        temp_mapping = {}
        for i, col in enumerate(data.columns, 1):
            standard_name = f'A{i}'
            temp_mapping[col] = column_mapping.get(standard_name, standard_name)
        return temp_mapping
    
    return {col: column_mapping.get(col, col) for col in data.columns}

def create_column_selector(data, label, key=None, multiselect=True, default=None, **kwargs):
    """Tworzy selector kolumn z użyciem proponowanych nazw"""
    columns = data.columns
    display_names = [get_display_column_name(col) for col in columns]
    
    if default is not None:
        # Konwersja domyślnych wartości na nazwy wyświetlane
        default = [get_display_column_name(col) for col in default]
    
    if multiselect:
        selected_display_names = st.multiselect(
            label,
            options=display_names,
            default=default,
            key=key,
            **kwargs
        )
        return [get_original_column_name(name) for name in selected_display_names]
    else:
        selected_display_name = st.selectbox(
            label,
            options=display_names,
            key=key,
            **kwargs
        )
        return get_original_column_name(selected_display_name)

# Funkcja do bezpiecznego wyświetlania DataFrame
def safe_display_dataframe(df, column_config=None, use_display_names=True):
    """
    Bezpiecznie wyświetla DataFrame w Streamlit z obsługą błędów i mapowaniem nazw kolumn.
    """
    try:
        # Kopia DataFrame do wyświetlenia
        df_display = df.copy()
        
        if use_display_names:
            # Mapowanie nazw kolumn na proponowane nazwy
            df_display.columns = [get_display_column_name(col) for col in df_display.columns]
        
        # Konwersja wszystkich kolumn na string
        df_display = df_display.astype(str)
        
        if column_config:
            return st.dataframe(df_display, column_config=column_config)
        return st.dataframe(df_display)
    except Exception as e:
        st.error(f"Błąd wyświetlania danych: {str(e)}")
        st.write("Dane w formie tekstowej:")
        st.write(df.to_string())

# Dodaj nowe funkcje na początku pliku, po importach
def display_missing_values(data):
    """Wyświetla szczegółowe informacje o brakujących wartościach"""
    missing_rows = data[data.isna().any(axis=1)]
    
    if missing_rows.empty:
        st.info("Brak wierszy z brakującymi wartościami.")
        return
        
    missing_cols = data.columns[data.isna().any()].tolist()
    
    st.write(f"Znaleziono {len(missing_rows)} wierszy z brakującymi wartościami w {len(missing_cols)} kolumnach.")
    
    # Podsumowanie brakujących wartości
    missing_summary = pd.DataFrame({
        'Kolumna': missing_cols,
        'Liczba brakujących': [data[col].isna().sum() for col in missing_cols],
        'Procent brakujących': [f"{(data[col].isna().sum() / len(data) * 100):.2f}%" for col in missing_cols]
    })
    
    st.write("Podsumowanie brakujących wartości per kolumna:")
    safe_display_dataframe(missing_summary, use_display_names=False)
    
    # Szczegóły brakujących wartości w wierszach
    st.write("\nSzczegóły wierszy z brakującymi wartościami:")
    
    # Zamiast zagnieżdżonych expanderów, używamy tabelki z wszystkimi informacjami
    missing_details = []
    for idx, row in missing_rows.iterrows():
        missing_in_row = row[row.isna()].index.tolist()
        for col in missing_in_row:
            missing_details.append({
                'Indeks wiersza': idx,
                'Kolumna': col,
                'Wartość': 'Brak'
            })
    
    if missing_details:
        details_df = pd.DataFrame(missing_details)
        st.dataframe(
            details_df,
            column_config={
                "Indeks wiersza": st.column_config.NumberColumn(width="small"),
                "Kolumna": st.column_config.TextColumn(width="medium"),
                "Wartość": st.column_config.TextColumn(width="small")
            }
        )

def create_editable_dataframe(data, start_idx, end_idx):
    """Tworzy edytowalny dataframe z zachowaniem oryginalnych indeksów"""
    display_data = data.iloc[start_idx:end_idx].copy()
    
    # Sprawdź czy kolumna 'ID' już istnieje i usuń ją jeśli tak
    if 'ID' in display_data.columns:
        display_data = display_data.drop(columns=['ID'])
    
    # Dodaj kolumnę z indeksami na początku, zaczynając od 0
    display_data.insert(0, 'ID', display_data.index)
    
    # Mapuj nazwy kolumn na przyjazne nazwy (oprócz kolumny indeksu)
    display_names = ['ID']
    for col in data.columns:
        if col != 'ID':  # Nie mapuj kolumny indeksu ponownie
            display_names.append(get_display_column_name(col))
    
    display_data.columns = display_names
    
    # Konfiguracja kolumn
    column_config = {
        'ID': st.column_config.NumberColumn(
            width='small',
            help='Identyfikator wiersza (od 0)',
            disabled=True
        )
    }
    
    # Dodaj konfigurację dla pozostałych kolumn
    column_config.update({
        col: st.column_config.Column(
            label=col,
            width="medium",
            required=True
        ) for col in display_names[1:]  # Pomijamy kolumnę ID
    })
    
    # Wyświetl edytowalną tabelę
    edited_df = st.data_editor(
        display_data,
        column_config=column_config,
        num_rows="dynamic",
        key=f"editor_{start_idx}_{end_idx}"
    )
    
    if not edited_df.equals(display_data):
        # Usuń kolumnę ID przed zapisem zmian
        edited_df = edited_df.drop(columns=['ID'])
        # Przywróć oryginalne nazwy kolumn
        edited_df.columns = data.columns
        
        # Zapisz zmiany w oryginalnych danych
        original_indices = data.iloc[start_idx:end_idx].index
        st.session_state.data.loc[original_indices] = edited_df
        st.success("Zmiany zostały zapisane!")
    
    return edited_df

def safe_paginated_display(data, rows_per_page, current_page):
    """Bezpieczne wyświetlanie danych z paginacją"""
    try:
        total_rows = len(data)
        start_idx = (current_page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_rows)
        
        create_editable_dataframe(data, start_idx, end_idx)
        
    except Exception as e:
        st.error(f"Błąd wyświetlania danych: {str(e)}")
        st.write("Spróbuj zmniejszyć liczbę wyświetlanych wierszy")

# Add after imports, before main code

def plot_parallel_coordinates(data, labels, features):
    """Tworzy wykres współrzędnych równoległych dla klastrów"""
    df_plot = pd.DataFrame(data, columns=features).copy()
    df_plot['Klaster'] = labels
    
    # Tworzenie wykresu
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df_plot['Klaster'].astype('category').cat.codes,  # Convert to numeric codes
                colorscale='Viridis'
            ),
            dimensions=[dict(
                range=[df_plot[feat].min(), df_plot[feat].max()],
                label=feat,
                values=df_plot[feat]
            ) for feat in features]
        )
    )
    
    # Dostosowanie layoutu
    fig.update_layout(
        title="Charakterystyka klastrów - wykres równoległych współrzędnych",
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def plot_3d_scatter(data, labels, features):
    """Tworzy trójwymiarowy wykres rozrzutu"""
    if len(features) >= 3:
        df_plot = pd.DataFrame({
            'x': data[features[0]],
            'y': data[features[1]],
            'z': data[features[2]],
            'Klaster': [f'Klaster {l}' for l in labels]
        })
        
        fig = px.scatter_3d(
            df_plot, 
            x='x', y='y', z='z',
            color='Klaster',
            labels={'x': features[0], 'y': features[1], 'z': features[2]},
            title="Wizualizacja klastrów 3D"
        )
        
        return fig
    return None

def plot_cluster_density(data, labels, feature):
    """Tworzy wykres gęstości dla wybranej cechy w klastrach"""
    df_plot = pd.DataFrame({
        'Wartość': data[feature],
        'Klaster': [f'Klaster {l}' for l in labels]
    })
    
    fig = px.violin(
        df_plot,
        x='Klaster',
        y='Wartość',
        box=True,
        title=f'Rozkład gęstości dla cechy {feature}'
    )
    return fig

# Ustawienia strony
st.set_page_config(
    page_title="Analiza zbioru Credit Approval",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Funkcja do cachowania danych w sesji
@st.cache_data
def get_data():
    if 'data' not in st.session_state:
        return None
    return st.session_state.data


# Inicjalizacja sesji
if 'data' not in st.session_state:
    st.session_state.data = None
    st.session_state.file_name = None
    st.session_state.page = 1

if 'clustering_state' not in st.session_state:
    st.session_state.clustering_state = {
        'viz_cols_2d': None,
        'viz_cols_3d': None,
        'show_visualizations': False,
        'metrics': None,
        'X_scaled': None,
        'X_original': None,
        'model': None
    }

# Tytuł aplikacji
st.title("Analiza zbioru Credit Approval")

# Sidebar - wczytywanie danych
with st.sidebar:
    st.header("Wczytywanie danych")

    upload_option = st.radio(
        "Wybierz sposób wczytania danych",
        ["Wczytaj plik CSV", "Użyj przykładowego zbioru danych"],
        index=1
    )

    if upload_option == "Wczytaj plik CSV":
        uploaded_file = st.file_uploader("Wybierz plik CSV", type=["csv"])

        if uploaded_file is not None:
            # Wczytanie pliku
            data = load_csv(uploaded_file)
            if data is not None:
                st.session_state.data = data
                st.session_state.file_name = uploaded_file.name
                st.success(f"Wczytano plik: {uploaded_file.name}")
            else:
                st.error("Błąd wczytywania pliku.")
    else:
        if st.button("Wczytaj przykładowy zbioru Credit Approval"):
            with st.spinner("Wczytywanie przykładowych danych..."):
                data = load_sample_data()
                if data is not None:
                    st.session_state.data = data
                    st.session_state.file_name = "credit_approval.csv"
                    st.success("Wczytano przykładowy zbiór danych.")
                else:
                    st.error("Błąd wczytywania przykładowych danych.")

    # Informacje o zbiorze danych
    if st.session_state.data is not None:
        st.subheader("Informacje o zbiorze danych")
        info = get_dataset_info(st.session_state.data)
        st.write(f"Liczba wierszy: {info['rows']}")
        st.write(f"Liczba kolumn: {info['columns']}")
        st.write(f"Brakujące wartości: {info['missing_values']}")
        st.write(f"Zduplikowane wiersze: {info['duplicated_rows']}")

        if st.checkbox("Pokaż typy danych"):
            dtypes_df = pd.DataFrame({'Typ danych': st.session_state.data.dtypes})
            safe_display_dataframe(dtypes_df)

        if st.checkbox("Pokaż korelacje"):
            corr_matrix = calculate_correlations(st.session_state.data)
            safe_display_dataframe(corr_matrix)

        if st.checkbox("Pokaż nazwy kolumn"):
            column_names = info['columns_names']
            display_names = [get_display_column_name(col) for col in column_names]
            
            # Utworzenie DataFrame z oryginalnymi i proponowanymi nazwami
            names_df = pd.DataFrame({
                'Oryg. kod': column_names,
                'Proponowana nazwa (PL)': display_names
            })
            
            # Wyświetlenie tabeli z formatowaniem
            safe_display_dataframe(
                names_df,
                column_config={
                    "Oryg. kod": st.column_config.TextColumn(width="small"),
                    "Proponowana nazwa (PL)": st.column_config.TextColumn(width="medium")
                }
            )

# Główny panel aplikacji
if st.session_state.data is not None:
    data = st.session_state.data

    # Wyświetlenie wszystkich danych z paginacją
    st.subheader("Podgląd danych")
    
    # Dodaj opcję filtrowania wierszy
    filter_rows = st.checkbox("Filtruj wiersze po indeksach")
    if filter_rows:
        indices_help = """
        Wprowadź indeksy wierszy w jednym z formatów:
        - Pojedyncze liczby: "1,3,5"
        - Zakresy: "1-5"
        - Kombinacje: "1,3-5,7,10-12"
        """
        indices_str = st.text_input("Indeksy wierszy:", help=indices_help)
        
        if indices_str:
            try:
                # Parsowanie indeksów
                indices = set()
                for part in indices_str.split(','):
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        indices.update(range(start, end + 1))
                    else:
                        indices.add(int(part))
                
                # Filtrowanie danych
                filtered_data = data.loc[sorted(indices)]
                total_rows = len(filtered_data)
                data_to_display = filtered_data
            except Exception as e:
                st.error(f"Błąd w formacie indeksów: {str(e)}")
                data_to_display = data
                total_rows = len(data)
        else:
            data_to_display = data
            total_rows = len(data)
    else:
        data_to_display = data
        total_rows = len(data)

    # Kontrolka do wyboru liczby wierszy na stronie
    rows_per_page = st.selectbox(
        "Liczba wierszy na stronie",
        options=[10, 20, 50, 100, 500, "Wszystkie"],
        index=0
    )
    
    # Obliczenie całkowitej liczby stron
    if rows_per_page == "Wszystkie":
        rows_per_page = total_rows
    else:
        rows_per_page = int(rows_per_page)
    
    total_pages = (total_rows + rows_per_page - 1) // rows_per_page
    
    # Użyj wartości z sesji zamiast zmiennej lokalnej
    if 'page' not in st.session_state:
        st.session_state.page = 1
    
    # Wybór strony
    if total_pages > 1:
        current_page = st.number_input(
            f"Strona (1-{total_pages})", 
            min_value=1, 
            max_value=total_pages, 
            value=st.session_state.page
        )
        st.session_state.page = current_page
    else:
        st.session_state.page = 1
    
    # Obliczenie zakresu wierszy do wyświetlenia
    start_idx = (st.session_state.page - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, total_rows)
    
    # Wyświetlenie informacji o zakresie
    st.write(f"Wyświetlanie wierszy {start_idx + 1}-{end_idx} z {total_rows}")
    
    # Modyfikacja funkcji create_editable_dataframe aby pokazywała indeksy
    def create_editable_dataframe(data, start_idx, end_idx):
        """Tworzy edytowalny dataframe z zachowaniem oryginalnych indeksów"""
        display_data = data.iloc[start_idx:end_idx].copy()
        
        # Sprawdź czy kolumna 'ID' już istnieje i usuń ją jeśli tak
        if 'ID' in display_data.columns:
            display_data = display_data.drop(columns=['ID'])
        
        # Dodaj kolumnę z indeksami na początku, zaczynając od 0
        display_data.insert(0, 'ID', display_data.index)
        
        # Mapuj nazwy kolumn na przyjazne nazwy (oprócz kolumny indeksu)
        display_names = ['ID']
        for col in data.columns:
            if col != 'ID':  # Nie mapuj kolumny indeksu ponownie
                display_names.append(get_display_column_name(col))
        
        display_data.columns = display_names
        
        # Konfiguracja kolumn
        column_config = {
            'ID': st.column_config.NumberColumn(
                width='small',
                help='Identyfikator wiersza (od 0)',
                disabled=True
            )
        }
        
        # Dodaj konfigurację dla pozostałych kolumn
        column_config.update({
            col: st.column_config.Column(
                label=col,
                width="medium",
                required=True
            ) for col in display_names[1:]  # Pomijamy kolumnę ID
        })
        
        # Wyświetl edytowalną tabelę
        edited_df = st.data_editor(
            display_data,
            column_config=column_config,
            num_rows="dynamic",
            key=f"editor_{start_idx}_{end_idx}"
        )
        
        if not edited_df.equals(display_data):
            # Usuń kolumnę ID przed zapisem zmian
            edited_df = edited_df.drop(columns=['ID'])
            # Przywróć oryginalne nazwy kolumn
            edited_df.columns = data.columns
            
            # Zapisz zmiany w oryginalnych danych
            original_indices = data.iloc[start_idx:end_idx].index
            st.session_state.data.loc[original_indices] = edited_df
            st.success("Zmiany zostały zapisane!")
        
        return edited_df

    # Wyświetlenie danych
    safe_paginated_display(data_to_display, rows_per_page, st.session_state.page)

    # Dodanie przycisków nawigacji
    if total_pages > 1:
        cols = st.columns(4)
        
        with cols[0]:
            if st.button("⏮️ Pierwsza", disabled=st.session_state.page==1):
                st.session_state.page = 1
                st.rerun()
                
        with cols[1]:
            if st.button("◀️ Poprzednia", disabled=st.session_state.page==1):
                st.session_state.page = max(1, st.session_state.page - 1)
                st.rerun()
                
        with cols[2]:
            if st.button("▶️ Następna", disabled=st.session_state.page==total_pages):
                st.session_state.page = min(total_pages, st.session_state.page + 1)
                st.rerun()
                
        with cols[3]:
            if st.button("⏭️ Ostatnia", disabled=st.session_state.page==total_pages):
                st.session_state.page = total_pages
                st.rerun()

    # Zakładki
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Statystyki",
        "🔧 Przetwarzanie danych",
        "📈 Wizualizacja",
        "🤖 Grupowanie"
    ])

    # Zakładka 1: Statystyki
    with tab1:
        st.header("Analiza statystyczna")
        
        # Dodaj na początku zakładki:
        st.subheader("Brakujące wartości")
        display_missing_values(data)

        st.subheader("Statystyki dla kolumn numerycznych")
        num_stats = calculate_numerical_stats(data)
        if num_stats is not None:
            safe_display_dataframe(num_stats)
        else:
            st.info("Brak kolumn numerycznych w zbiorze danych.")

        st.subheader("Statystyki dla kolumn kategorycznych")
        cat_stats = calculate_categorical_stats(data)
        if cat_stats is not None:
            for col, stats in cat_stats.items():
                display_name = get_display_column_name(col)
                with st.expander(f"**{display_name}**"):
                    st.write(f"Unikalne wartości: {stats['Unikalne wartości']}")
                    st.write(f"Moda: {stats['Moda']}")
                    st.write(f"Liczba wystąpień mody: {stats['Liczba wystąpień mody']}")
                    st.write(f"Brakujące wartości: {stats['Brakujące wartości']}")
                    st.write("Najczęstsze wartości:")
                    for val, count in stats['Najczęstsze 5 wartości'].items():
                        st.write(f"- {val}: {count}")
        else:
            st.info("Brak kolumn kategorycznych w zbiorze danych.")

        st.subheader("Korelacje między atrybutami")
        correlation_method = st.selectbox(
            "Wybierz metodę korelacji",
            ["pearson", "kendall", "spearman"],
            index=0
        )

        corr_matrix = calculate_correlations(data, method=correlation_method)
        if corr_matrix is not None:
            safe_display_dataframe(corr_matrix)
        else:
            st.info("Brak kolumn numerycznych do obliczenia korelacji.")

    # Zakładka 2: Przetwarzanie danych
    with tab2:
        st.header("Przetwarzanie danych")

        # Przycisk do przywracania oryginalnych danych
        if st.button("Przywróć oryginalne dane"):
            if st.session_state.file_name == "credit_approval.csv":
                st.session_state.data = load_sample_data()
            st.rerun()

        st.write("Bieżący rozmiar danych:", data.shape)

        processing_option = st.radio(
            "Wybierz operację na danych",
            ["Ekstrakcja/usuwanie wierszy", "Zamiana wartości", "Obsługa brakujących danych",
             "Usuwanie duplikatów", "Skalowanie danych", "Kodowanie zmiennych kategorycznych"]
        )

        if processing_option == "Ekstrakcja/usuwanie wierszy":
            st.subheader("Ekstrakcja lub usuwanie wierszy")
            
            operation_mode = st.radio(
                "Wybierz tryb operacji",
                ["Zachowaj wybrane wiersze", "Usuń wybrane wiersze"]
            )
            
            input_method = st.radio(
                "Sposób wyboru wierszy",
                ["Po indeksach", "Po wartościach w kolumnie"]
            )

            if input_method == "Po indeksach":
                indices_help = """
                Wprowadź indeksy wierszy w jednym z formatów:
                - Pojedyncze liczby: "1,3,5"
                - Zakresy: "1-5"
                - Kombinacje: "1,3-5,7,10-12"
                """
                indices_str = st.text_input("Indeksy wierszy:", help=indices_help)
                
                if st.button("Wykonaj operację"):
                    if indices_str:
                        mode = 'keep' if operation_mode == "Zachowaj wybrane wiersze" else 'remove'
                        st.session_state.data = select_rows(st.session_state.data, indices_str, mode=mode)
                        st.success(f"Operacja zakończona pomyślnie")
                        st.rerun()
                    else:
                        st.warning("Wprowadź indeksy wierszy")

            else:  # Po wartościach w kolumnie
                col = create_column_selector(data, "Wybierz kolumnę", multiselect=False)
                value = st.text_input("Podaj wartość do wyszukania")
                
                if st.button("Wykonaj operację"):
                    if value:
                        mode = 'keep' if operation_mode == "Zachowaj wybrane wiersze" else 'remove'
                        mask = data[col].astype(str) == str(value)
                        indices = data[mask].index.tolist()
                        st.session_state.data = select_rows(st.session_state.data, 
                                                          ','.join(map(str, indices)), 
                                                          mode=mode)
                        st.success(f"Operacja zakończona pomyślnie")
                        st.rerun()
                    else:
                        st.warning("Wprowadź wartość do wyszukania")

        elif processing_option == "Zamiana wartości":
            st.subheader("Zamiana wartości w kolumnach")
            
            replacement_mode = st.radio(
                "Tryb zamiany",
                ["Pojedyncza zamiana", "Wiele zamian"]
            )
            
            if replacement_mode == "Pojedyncza zamiana":
                col = create_column_selector(data, "Wybierz kolumnę", multiselect=False)
                old_value = st.text_input("Stara wartość")
                new_value = st.text_input("Nowa wartość")
                
                if st.button("Zamień wartości"):
                    if old_value or old_value == '':
                        replacements = [(col, old_value, new_value)]
                        st.session_state.data = replace_values_in_columns(data, replacements)
                        st.success("Zamiana zakończona pomyślnie")
                        st.rerun()
                    else:
                        st.warning("Wprowadź starą wartość")
            
            else:  # Wiele zamian
                st.write("Wprowadź pary wartości do zamiany")
                
                num_replacements = st.number_input("Liczba zamian", min_value=1, max_value=10, value=1)
                replacements = []
                
                for i in range(num_replacements):
                    st.write(f"Zamiana {i+1}")
                    col = create_column_selector(data, f"Kolumna {i+1}", multiselect=False, key=f"col_{i}")
                    old_val = st.text_input(f"Stara wartość {i+1}", key=f"old_{i}")
                    new_val = st.text_input(f"Nowa wartość {i+1}", key=f"new_{i}")
                    
                    if col and (old_val or old_val == ''):
                        replacements.append((col, old_val, new_val))
                
                if st.button("Wykonaj zamiany"):
                    if replacements:
                        st.session_state.data = replace_values_in_columns(data, replacements)
                        st.success("Zamiany zakończone pomyślnie")
                        st.rerun()
                    else:
                        st.warning("Wprowadź przynajmniej jedną zamianę")

        elif processing_option == "Obsługa brakujących danych":
            st.subheader("Obsługa brakujących danych")

            na_columns = data.columns[data.isna().any()].tolist()
            if not na_columns:
                st.info("Brak kolumn z brakującymi wartościami.")
            else:
                st.write("Kolumny z brakującymi wartościami:")
                na_counts = data[na_columns].isna().sum()
                for col, count in na_counts.items():
                    display_name = get_display_column_name(col)
                    st.write(f"- {display_name}: {count} brakujących wartości")

                handling_method = st.radio(
                    "Wybierz metodę obsługi brakujących danych",
                    ["Usuń wiersze", "Usuń kolumny", "Wypełnij wartościami"]
                )

                target_columns = create_column_selector(
                    data[na_columns], 
                    "Wybierz kolumny do przetworzenia (puste = wszystkie z brakującymi)",
                    multiselect=True
                )

                if handling_method == "Usuń wiersze":
                    if st.button("Usuń wiersze z brakującymi wartościami"):
                        st.session_state.data = handle_missing_values(
                            data,
                            method='drop_rows',
                            columns=target_columns if target_columns else na_columns
                        )
                        st.success("Usunięto wiersze z brakującymi wartościami.")
                        st.rerun()

                elif handling_method == "Usuń kolumny":
                    if st.button("Usuń kolumny z brakującymi wartościami"):
                        st.session_state.data = handle_missing_values(
                            data,
                            method='drop_columns',
                            columns=target_columns if target_columns else na_columns
                        )
                        st.success("Usunięto kolumny z brakującymi wartościami.")
                        st.rerun()

                else:  # Wypełnij wartościami
                    fill_method = st.radio(
                        "Wybierz metodę wypełniania",
                        ["mean", "median", "mode", "zero"]
                    )

                    if st.button("Wypełnij brakujące wartości"):
                        st.session_state.data = handle_missing_values(
                            data,
                            method=fill_method,
                            columns=target_columns if target_columns else na_columns
                        )
                        st.success(f"Wypełniono brakujące wartości metodą: {fill_method}")
                        st.rerun()

        elif processing_option == "Usuwanie duplikatów":
            st.subheader("Usuwanie duplikatów")

            dup_count = data.duplicated().sum()
            st.write(f"Liczba zduplikowanych wierszy: {dup_count}")

            if dup_count > 0:
                if st.button("Usuń duplikaty"):
                    st.session_state.data = remove_duplicates(data)
                    st.success(f"Usunięto {dup_count} zduplikowanych wierszy.")
                    st.rerun()
            else:
                st.info("Brak zduplikowanych wierszy w zbiorze danych.")

        elif processing_option == "Skalowanie danych":
            st.subheader("Skalowanie danych")

            # Tylko kolumny numeryczne
            num_cols = data.select_dtypes(include=['number']).columns.tolist()

            if not num_cols:
                st.info("Brak kolumn numerycznych do skalowania.")
            else:
                scale_method = st.radio(
                    "Wybierz metodę skalowania",
                    ["minmax", "standard"]
                )

                cols_to_scale = create_column_selector(
                    data.select_dtypes(include=['number']), 
                    "Wybierz kolumny do skalowania",
                    multiselect=True
                )

                if st.button("Skaluj dane"):
                    if cols_to_scale:
                        st.session_state.data = scale_data(data, cols_to_scale, method=scale_method)
                        st.success(f"Przeskalowano kolumny: {', '.join(cols_to_scale)}")
                        st.rerun()
                    else:
                        st.warning("Nie wybrano kolumn do skalowania.")

        elif processing_option == "Kodowanie zmiennych kategorycznych":
            st.subheader("Kodowanie zmiennych kategorycznych")

            # Tylko kolumny kategoryczne
            cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

            if not cat_cols:
                st.info("Brak kolumn kategorycznych do kodowania.")
            else:
                encoding_method = st.radio(
                    "Wybierz metodę kodowania",
                    ["onehot", "binary"]
                )

                cols_to_encode = create_column_selector(
                    data.select_dtypes(include=['object', 'category']), 
                    "Wybierz kolumny do kodowania",
                    multiselect=True
                )

                if st.button("Koduj dane"):
                    if cols_to_encode:
                        st.session_state.data = encode_categorical(data, cols_to_encode, method=encoding_method)
                        st.success(f"Zakodowano kolumny: {', '.join(cols_to_encode)}")
                        st.rerun()
                    else:
                        st.warning("Nie wybrano kolumn do kodowania.")

    # Zakładka 3: Wizualizacja
    with tab3:
        st.header("Wizualizacja danych")

        viz_type = st.selectbox(
            "Wybierz typ wykresu",
            ["Histogram", "Wykres pudełkowy", "Wykres punktowy",
             "Wykres słupkowy", "Wykres kołowy", "Wykres par"]
        )

        if viz_type == "Histogram":
            st.subheader("Histogram")
            st.markdown("""
            **Co reprezentuje**: Rozkład wartości w wybranej kolumnie numerycznej.
            
            **Jak czytać**:
            - Wysokość słupka pokazuje częstość występowania wartości
            - Szerokość słupka to zakres wartości (przedział)
            - Kształt histogramu sugeruje rodzaj rozkładu (np. normalny, skośny)
            """)

            # Tylko kolumny numeryczne
            num_cols = data.select_dtypes(include=['number']).columns.tolist()

            if not num_cols:
                st.info("Brak kolumn numerycznych dla histogramu.")
            else:
                col = create_column_selector(
                    data.select_dtypes(include=['number']), 
                    "Wybierz kolumnę",
                    multiselect=False,
                    key="histogram_col"
                )
                bins = st.slider("Liczba przedziałów", min_value=5, max_value=100, value=20)

                fig = histogram(data, col, bins=bins)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, key="histogram_plot")

        elif viz_type == "Wykres pudełkowy":
            st.subheader("Wykres pudełkowy")
            st.markdown("""
            **Co reprezentuje**: Rozkład i statystyki wartości numerycznych.
            
            **Jak czytać**:
            - Środkowa linia = mediana
            - Dolna i górna krawędź pudełka = pierwszy i trzeci kwartyl
            - Wąsy = minimum i maksimum (bez wartości odstających)
            - Punkty poza wąsami = wartości odstające
            """)

            # Tylko kolumny numeryczne
            num_cols = data.select_dtypes(include=['number']).columns.tolist()

            if not num_cols:
                st.info("Brak kolumn numerycznych dla wykresu pudełkowego.")
            else:
                col = create_column_selector(
                    data.select_dtypes(include=['number']), 
                    "Wybierz kolumnę",
                    multiselect=False
                )

                fig = box_plot(data, col)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, key="boxplot_plot")

        elif viz_type == "Wykres punktowy":
            st.subheader("Wykres punktowy")
            st.markdown("""
            **Co reprezentuje**: Zależność między dwiema zmiennymi numerycznymi.
            
            **Jak czytać**:
            - Każdy punkt reprezentuje jedną obserwację
            - Położenie punktu pokazuje wartości dla obu zmiennych
            - Skupiska punktów sugerują korelację
            - Kolory mogą reprezentować dodatkową zmienną
            """)

            # Tylko kolumny numeryczne
            num_cols = data.select_dtypes(include=['number']).columns.tolist()

            if len(num_cols) < 2:
                st.info("Potrzebne są co najmniej 2 kolumny numeryczne dla wykresu punktowego.")
            else:
                x_col = create_column_selector(
                    data.select_dtypes(include=['number']), 
                    "Wybierz kolumnę dla osi X",
                    multiselect=False,
                    key="scatter_x"
                )
                y_col = create_column_selector(
                    data.select_dtypes(include=['number']), 
                    "Wybierz kolumnę dla osi Y",
                    multiselect=False,
                    key="scatter_y"
                )

                # Opcjonalnie dodaj kolumnę koloryzowania
                color_options = ["Brak"] + data.columns.tolist()
                color_col = st.selectbox("Koloruj według (opcjonalnie)", color_options)
                color_col = None if color_col == "Brak" else color_col

                fig = scatter_plot(data, x_col, y_col, color_column=color_col)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, key="scatter_plot")

        elif viz_type == "Wykres słupkowy":
            st.subheader("Wykres słupkowy")
            st.markdown("""
            **Co reprezentuje**: 
            - Tryb "Liczności": Liczebność kategorii w wybranej kolumnie
            - Tryb "Wartości": Zależność między zmienną kategoryczną a numeryczną
            
            **Jak czytać**:
            - Wysokość słupka pokazuje wartość lub liczebność
            - Szerokość słupków jest stała
            - Etykiety na osi X to kategorie
            """)

            chart_type = st.radio("Typ wykresu słupkowego", ["Liczności", "Wartości"])

            if chart_type == "Liczności":
                col = create_column_selector(data, "Wybierz kolumnę", multiselect=False)
                fig = bar_chart(data, col)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, key="bar_plot")
            else:
                # Wybierz kolumnę kategoryczną dla osi X
                x_options = data.select_dtypes(include=['object', 'category']).columns.tolist()
                if not x_options:
                    st.info("Brak kolumn kategorycznych dla osi X.")
                else:
                    x_col = create_column_selector(
                        data.select_dtypes(include=['object', 'category']), 
                        "Wybierz kolumnę kategoryczną dla osi X",
                        multiselect=False,
                        key="bar_x"
                    )

                    # Wybierz kolumnę numeryczną dla osi Y
                    y_options = data.select_dtypes(include=['number']).columns.tolist()
                    if not y_options:
                        st.info("Brak kolumn numerycznych dla osi Y.")
                    else:
                        y_col = create_column_selector(
                            data.select_dtypes(include=['number']), 
                            "Wybierz kolumnę numeryczną dla osi Y",
                            multiselect=False,
                            key="bar_y"
                        )

                        fig = bar_chart(data, x_col, y_col)
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True, key="bar_plot")

        elif viz_type == "Wykres kołowy":
            st.subheader("Wykres kołowy")
            st.markdown("""
            **Co reprezentuje**: Udział poszczególnych kategorii w całości (procentowy).
            
            **Jak czytać**:
            - Wielkość wycinków pokazuje proporcje kategorii
            - Procenty sumują się do 100%
            - Kolory rozróżniają kategorie
            - Najlepszy dla małej liczby kategorii (maks. 6-8)
            """)

            # Lista kolumn do wykluczenia
            excluded_columns = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']
            
            # Filtrowanie kolumn
            filtered_data = data.drop(columns=excluded_columns, errors='ignore')

            # Preferowane kolumny kategoryczne, ale można użyć dowolnej
            col = create_column_selector(
                filtered_data, 
                "Wybierz kolumnę", 
                multiselect=False,
                key="pie_chart_column"
            )

            fig = pie_chart(data, col)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, key="pie_plot")

        elif viz_type == "Wykres par":
            st.subheader("Wykres par")
            st.markdown("""
            **Co reprezentuje**: Wzajemne relacje między wieloma zmiennymi numerycznymi.
            
            **Jak czytać**:
            - Każde pole to osobny wykres punktowy
            - Przekątna pokazuje rozkład pojedynczej zmiennej
            - Pola poza przekątną pokazują zależności między parami zmiennych
            - Kolory mogą reprezentować dodatkową zmienną kategoryczną
            """)

            # Tylko kolumny numeryczne
            num_cols = data.select_dtypes(include=['number']).columns.tolist()

            if len(num_cols) < 2:
                st.info("Potrzebne są co najmniej 2 kolumny numeryczne dla wykresu par.")
            else:
                sel_cols = create_column_selector(
                    data.select_dtypes(include=['number']), 
                    "Wybierz kolumny (maks. 5 zalecane)",
                    multiselect=True,
                    default=num_cols[:min(3, len(num_cols))]
                )

                # Opcjonalnie dodaj kolumnę koloryzowania
                color_options = ["Brak"] + data.select_dtypes(include=['object', 'category']).columns.tolist()
                hue = st.selectbox("Koloruj według (opcjonalnie)", color_options)
                hue = None if hue == "Brak" else hue

                if sel_cols and len(sel_cols) >= 2:
                    fig = pair_plot(data, columns=sel_cols, hue=hue)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True, key="pair_plot")
                else:
                    st.warning("Wybierz co najmniej 2 kolumny.")

    # Zakładka 4: Modelowanie
    with tab4:
        st.header("Grupowanie danych")
        st.write("""
        ### K-means Clustering
        K-means to algorytm grupowania, który dzieli dane na k grup (klastrów) na podstawie podobieństwa cech.
        Każdy klaster jest reprezentowany przez swój centroid (średni punkt).
        """)
        
        # Wybór kolumn do grupowania
        st.subheader("Wybór danych")
        clustering_cols = create_column_selector(
            data,
            "Wybierz kolumny do grupowania (tylko numeryczne będą użyte)",
            multiselect=True,
            key="clustering_columns"
        )

        # Parametry grupowania
        with st.expander("Parametry grupowania"):
            col1, col2 = st.columns(2)
            with col1:
                n_clusters = st.slider(
                    "Liczba klastrów (k)", 
                    min_value=2, 
                    max_value=10, 
                    value=3
                )
                init = st.selectbox(
                    "Metoda inicjalizacji",
                    options=['k-means++', 'random'],
                    index=0
                )
                max_iter = st.number_input(
                    "Maksymalna liczba iteracji",
                    min_value=100,
                    max_value=1000,
                    value=300,
                    step=50
                )
            
            with col2:
                n_init = st.number_input(
                    "Liczba inicjalizacji",
                    min_value=1,
                    max_value=20,
                    value=10,
                    step=1
                )
                random_state = st.number_input(
                    "Ziarno losowości",
                    value=42
                )

        # Przycisk do uruchomienia grupowania
        if st.button("Wykonaj grupowanie"):
            with st.spinner("Grupowanie danych..."):
                    X_scaled, X_original = prepare_data_for_clustering(data, clustering_cols)

                    if X_scaled is None:
                        st.error("Błąd przygotowania danych. Sprawdź, czy wybrane kolumny są odpowiednie.")
                    else:
                        model = KMeans(
                            n_clusters=n_clusters,
                            init=init,
                            max_iter=max_iter,
                            n_init=n_init,
                            random_state=random_state
                        )
                        model.fit(X_scaled)
                        
                        metrics = evaluate_clustering(X_scaled, model)
                        
                        if metrics is not None:
                            st.success("Grupowanie zakończone pomyślnie!")
                            
                            # Metryki
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Inertia", f"{metrics['inertia']:.2f}")
                            with col2:
                                st.metric("Współczynnik sylwetki", f"{metrics['silhouette']:.3f}")
                            
                            # Rozmiary klastrów
                            st.subheader("Rozmiary klastrów")
                            sizes_df = pd.DataFrame.from_dict(
                                metrics['cluster_sizes'], 
                                orient='index',
                                columns=['Liczba próbek']
                            )
                            safe_display_dataframe(sizes_df)
                            
                            # Centroidy
                            st.subheader("Centroidy klastrów")
                            with st.expander("Pokaż współrzędne centroidów"):
                                centroids_df = pd.DataFrame(
                                    metrics['centroids'],
                                    columns=X_scaled.columns,
                                    index=[f"Klaster {i}" for i in range(n_clusters)]
                                )
                                safe_display_dataframe(centroids_df)
                            

                            # Wizualizacje
                            st.subheader("Wizualizacje")
                            
                            # Wykres elbow method
                            st.write("#### Wykres łokcia (elbow method)")
                            st.write("""
                            Ten wykres pomaga w wyborze optymalnej liczby klastrów. 
                            Punkt 'zgięcia' (łokcia) sugeruje optymalną liczbę klastrów.
                            """)

                            # Obliczenia dla wykresu łokcia
                            k_range = range(2, min(11, len(X_scaled)))
                            inertias = []
                            with st.spinner("Obliczanie wykresu łokcia..."):
                                for k in k_range:
                                    kmeans = KMeans(n_clusters=k, random_state=42)
                                    kmeans.fit(X_scaled)
                                    inertias.append(kmeans.inertia_)

                            # Wyświetl wykres łokcia
                            fig_elbow = px.line(
                                x=list(k_range), 
                                y=inertias,
                                title="Metoda łokcia dla wyboru optymalnej liczby klastrów",
                                labels={'x': 'Liczba klastrów (k)', 'y': 'Inertia'}
                            )
                            fig_elbow.add_scatter(
                                x=[n_clusters], 
                                y=[model.inertia_], 
                                mode='markers',
                                marker=dict(size=10, color='red'),
                                name='Wybrana liczba klastrów'
                            )
                            st.plotly_chart(fig_elbow, use_container_width=True, key="elbow_plot_main")

                            # Wizualizacja klastrów 2D
                            if len(X_scaled.columns) >= 2:
                                st.write("#### Wizualizacja klastrów 2D")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    feat1 = st.selectbox(
                                        "Wybierz pierwszą cechę",
                                        options=X_scaled.columns,
                                        index=0,
                                        key='feat1_2d_select_main'
                                    )
                                
                                with col2:
                                    remaining_cols = [col for col in X_scaled.columns if col != feat1]
                                    feat2 = st.selectbox(
                                        "Wybierz drugą cechę",
                                        options=remaining_cols,
                                        index=0,
                                        key='feat2_2d_select_main'
                                    )
                                
                                # Generowanie wykresu
                                fig = plot_clusters_2d(X_scaled, metrics['labels'], metrics['centroids'], [feat1, feat2])
                                st.plotly_chart(fig, use_container_width=True, key="clustering_2d_plot_main")

                            # Wykres równoległych współrzędnych
                            st.write("#### Wykres równoległych współrzędnych")
                            st.write("""
                            Ten wykres pokazuje charakterystykę klastrów na wszystkich wymiarach jednocześnie.
                            Każda linia reprezentuje jeden klaster, a jej przebieg pokazuje wartości na poszczególnych osiach.
                            """)
                            fig_parallel = plot_parallel_coordinates(X_scaled, metrics['labels'], X_scaled.columns)
                            st.plotly_chart(fig_parallel, use_container_width=True, key="parallel_coords_plot_main")

                            # Wizualizacja 3D
                            if len(X_scaled.columns) >= 3:
                                st.write("#### Wizualizacja 3D")
                                st.write("""
                                Ten wykres pokazuje rozmieszczenie punktów w przestrzeni trójwymiarowej.
                                Możesz obracać wykres i oglądać klastry z różnych perspektyw.
                                """)
                                
                                viz_cols_3d = st.multiselect(
                                    "Wybierz 3 cechy do wizualizacji 3D",
                                    options=X_scaled.columns,
                                    default=list(X_scaled.columns[:3]),
                                    key="viz_3d_main"
                                )
                                
                                if len(viz_cols_3d) == 3:
                                    fig_3d = plot_3d_scatter(X_scaled, metrics['labels'], viz_cols_3d)
                                    st.plotly_chart(fig_3d, use_container_width=True, key="scatter_3d_plot_main")

                            # Wykresy gęstości
                            st.write("#### Rozkłady gęstości cech w klastrach")
                            st.write("""
                            Te wykresy pokazują rozkład wartości cech w poszczególnych klastrach.
                            Szerokość wykresu odpowiada częstości występowania danej wartości.
                            """)

                            for i, feature in enumerate(X_scaled.columns):
                                fig_density = plot_cluster_density(X_scaled, metrics['labels'], feature)
                                st.plotly_chart(fig_density, use_container_width=True, key=f"density_plot_main_{feature}_{i}")
