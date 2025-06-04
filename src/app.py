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
import importlib
import sys

if 'data_processing' in sys.modules:
    importlib.reload(sys.modules['data_processing'])

# Import modułów aplikacji
from data_loader import (
    load_csv, get_dataset_info, detect_column_types, 
    suggest_data_preprocessing, validate_csv_structure
)

from statisticss import (
    calculate_numerical_stats, calculate_categorical_stats, 
    calculate_correlations
)

from data_processing import (
    remove_rows_by_indices, keep_rows_by_indices,
    remove_rows_by_value, keep_rows_by_value,
    remove_columns, replace_values,
    handle_missing_values, remove_duplicates,
    scale_data, encode_categorical
)

from visualization import (
    histogram, box_plot, scatter_plot, bar_chart, pie_chart, pair_plot
)

def create_column_selector(data, label, key=None, multiselect=True, default=None, **kwargs):
    """Tworzy selector kolumn."""
    if data is None or data.empty:
        return [] if multiselect else None
        
    columns = data.columns.tolist()
    
    if multiselect:
        selected_columns = st.multiselect(
            label,
            options=columns,
            default=default,
            key=key,
            **kwargs
        )
        return selected_columns
    else:
        # Dla selectbox dodaj opcję None na początku jeśli nie ma default
        options = columns if default is not None else [None] + columns
        selected_column = st.selectbox(
            label,
            options=options,
            index=0 if default is None else None,
            key=key,
            **kwargs
        )
        return selected_column

def safe_display_dataframe(df, column_config=None, use_container_width=True):
    """Bezpiecznie wyświetla DataFrame w Streamlit."""
    try:
        if column_config:
            return st.dataframe(df, column_config=column_config, use_container_width=use_container_width)
        return st.dataframe(df, use_container_width=use_container_width)
    except Exception as e:
        st.error(f"Błąd wyświetlania danych: {str(e)}")
        st.write("Dane w formie tekstowej:")
        st.text(str(df))

def display_missing_values(data):
    """Wyświetla szczegółowe informacje o brakujących wartościach"""
    missing_rows = data[data.isna().any(axis=1)]
    
    if missing_rows.empty:
        st.info("✅ Brak wierszy z brakującymi wartościami.")
        return
        
    missing_cols = data.columns[data.isna().any()].tolist()
    
    st.write(f"❗ Znaleziono **{len(missing_rows)}** wierszy z brakującymi wartościami w **{len(missing_cols)}** kolumnach.")
    
    # Podsumowanie brakujących wartości
    missing_summary = []
    for col in missing_cols:
        missing_count = data[col].isna().sum()
        missing_percent = (missing_count / len(data)) * 100
        missing_summary.append({
            'Kolumna': col,
            'Liczba brakujących': missing_count,
            'Procent brakujących': f"{missing_percent:.1f}%"
        })
    
    st.write("**Podsumowanie brakujących wartości:**")
    safe_display_dataframe(pd.DataFrame(missing_summary))
    
    # Opcjonalnie pokaż szczegóły
    if st.checkbox("🔍 Pokaż szczegóły brakujących wartości", key="show_missing_details"):
        st.write("**Wiersze z brakującymi wartościami:**")
        
        # Ograniczymy wyświetlanie do pierwszych 100 wierszy z problemami
        display_missing = missing_rows.head(100)
        
        # Podświetl brakujące wartości
        def highlight_missing(val):
            return 'background-color: #ffcccc' if pd.isna(val) else ''
        
        styled_df = display_missing.style.applymap(highlight_missing)
        st.dataframe(styled_df, use_container_width=True)
        
        if len(missing_rows) > 100:
            st.info(f"Pokazano pierwsze 100 z {len(missing_rows)} wierszy z brakującymi wartościami")

def create_editable_dataframe(data, start_idx, end_idx):
    """Tworzy edytowalny dataframe."""
    display_data = data.iloc[start_idx:end_idx].copy()
    
    # Dodaj kolumnę z indeksami na początku
    display_data.insert(0, 'Indeks', display_data.index)
    
    # Konfiguracja kolumn
    column_config = {
        'Indeks': st.column_config.NumberColumn(
            width='small',
            help='Numer wiersza w zbiorze danych',
            disabled=True
        )
    }
    
    # Dodaj konfigurację dla pozostałych kolumn
    for col in data.columns:
        column_config[col] = st.column_config.Column(
            label=col,
            width="medium"
        )
    
    # Stabilny klucz bazujący na zakresie
    unique_key = f"editor_{start_idx}_{end_idx}"
    
    # Wyświetl edytowalną tabelę
    edited_df = st.data_editor(
        display_data,
        column_config=column_config,
        num_rows="dynamic",
        key=unique_key,
        use_container_width=True
    )
    
    # Sprawdź czy były zmiany (porównaj bez kolumny Indeks)
    display_without_index = display_data.drop(columns=['Indeks'])
    edited_without_index = edited_df.drop(columns=['Indeks']) if 'Indeks' in edited_df.columns else edited_df
    
    if not edited_without_index.equals(display_without_index):
        # Zapisz zmiany w oryginalnych danych
        original_indices = data.iloc[start_idx:end_idx].index
        st.session_state.data.loc[original_indices] = edited_without_index.values
        st.success("✅ Zmiany zostały zapisane!")
    
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
        st.info("Spróbuj zmniejszyć liczbę wyświetlanych wierszy")

# Ustawienia strony
st.set_page_config(
    page_title="Analiza zbioru Credit Approval",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inicjalizacja sesji
if 'data' not in st.session_state:
    st.session_state.data = None
    st.session_state.file_name = None
    st.session_state.page = 1

# Tytuł aplikacji
st.title("Analiza zbioru Credit Approval")

# Sidebar - wczytywanie danych
with st.sidebar:
    st.header("📁 Wczytywanie danych")
    st.markdown("Wybierz plik CSV do analizy")

    uploaded_file = st.file_uploader(
        "Wybierz plik CSV", 
        type=["csv"],
        help="Obsługiwane formaty: CSV z różnymi separatorami (,;|\\t)"
    )

    if uploaded_file is not None:
        with st.spinner("🔄 Wczytywanie i analizowanie pliku..."):
            # Wczytanie pliku
            data = load_csv(uploaded_file)
            
            if data is not None:
                # Walidacja struktury
                validation = validate_csv_structure(data)
                
                if validation['is_valid']:
                    # Zapisz dane do sesji
                    st.session_state.data = data
                    st.session_state.file_name = uploaded_file.name
                    
                    st.success(f"✅ Plik wczytany pomyślnie!")
                    
                    # Wyświetl błędy jeśli są
                    if validation['errors']:
                        st.error("❌ Błędy:")
                        for error in validation['errors']:
                            st.write(f"• {error}")
                    
                    # Wyświetl ostrzeżenia jeśli są
                    if validation['warnings']:
                        st.warning("⚠️ Ostrzeżenia:")
                        for warning in validation['warnings']:
                            st.write(f"• {warning}")
                    
                    # Wyświetl informacje dodatkowe
                    if validation['info']:
                        with st.expander("ℹ️ Dodatkowe informacje"):
                            for info in validation['info']:
                                st.write(f"• {info}")
                else:
                    st.error("❌ Plik zawiera błędy krytyczne i nie może być wczytany")
                    for error in validation['errors']:
                        st.write(f"• {error}")
    
    else:
        st.info("👆 Wczytaj plik CSV, aby rozpocząć analizę")
        
        # Przykłady formatów CSV
        with st.expander("📋 Obsługiwane formaty CSV"):
            st.markdown("""
            **Separatory:** `,` `;` `|` `\\t` (tab)
            
            **Kodowanie:** UTF-8, Latin-1, CP1250, ISO-8859-1
            
            **Brakujące wartości:** `?` `NA` `N/A` `null` (puste komórki)
            
            **Przykład poprawnego pliku:**
            ```
            nazwa,wiek,miasto
            Jan,25,Warszawa
            Anna,30,Kraków
            ```
            """)

    # Informacje o zbiorze danych (tylko jeśli dane są wczytane)
    if st.session_state.data is not None:
        st.divider()
        st.subheader("📊 Informacje o danych")
        
        data = st.session_state.data
        info = get_dataset_info(data)
        
        # Podstawowe metryki w kolumnach
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📏 Wiersze", info['rows'])
            st.metric("❓ Brakujące", info['missing_values'])
        with col2:
            st.metric("🔢 Kolumny", info['columns'])
            st.metric("🔄 Duplikaty", info['duplicated_rows'])
        
        # Rozmiar w pamięci
        st.metric("💾 Rozmiar", f"{info['memory_usage']:.2f} MB")
        
        # Reset danych
        if st.button("🔄 Wyczyść dane", help="Usuń wczytane dane i zacznij od nowa"):
            st.session_state.data = None
            st.session_state.file_name = None
            st.rerun()

# Główny panel aplikacji
if st.session_state.data is not None:
    data = st.session_state.data

    # Nagłówek z informacjami o pliku
    st.header(f"📊 Analiza danych: {st.session_state.file_name}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Wiersze", data.shape[0])
    with col2:
        st.metric("Kolumny", data.shape[1])
    with col3:
        missing_count = data.isnull().sum().sum()
        st.metric("Brakujące", missing_count)
    with col4:
        duplicate_count = data.duplicated().sum()
        st.metric("Duplikaty", duplicate_count)

    # Wyświetlenie danych z paginacją
    st.subheader("🔍 Podgląd i edycja danych")
    
    # Opcje filtrowania
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Filtrowanie wierszy
        filter_rows = st.checkbox("🎯 Filtruj wiersze po indeksach")
        if filter_rows:
            indices_help = """
            Wprowadź indeksy wierszy w jednym z formatów:
            - Pojedyncze liczby: "1,3,5"
            - Zakresy: "1-5"
            - Kombinacje: "1,3-5,7,10-12"
            """
            indices_str = st.text_input("Indeksy wierszy:", help=indices_help, key="filter_indices")
            
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
                    valid_indices = [i for i in indices if i < len(data)]
                    if valid_indices:
                        filtered_data = data.loc[valid_indices]
                        total_rows = len(filtered_data)
                        data_to_display = filtered_data
                        st.success(f"✅ Filtrowanie: {len(valid_indices)} wierszy")
                    else:
                        st.warning("⚠️ Brak prawidłowych indeksów")
                        data_to_display = data
                        total_rows = len(data)
                except Exception as e:
                    st.error(f"❌ Błąd w formacie indeksów: {str(e)}")
                    data_to_display = data
                    total_rows = len(data)
            else:
                data_to_display = data
                total_rows = len(data)
        else:
            data_to_display = data
            total_rows = len(data)
    
    with col2:
        # Kontrolka liczby wierszy na stronie
        rows_per_page = st.selectbox(
            "Wierszy na stronie",
            options=[5, 10, 20, 50, 100, "Wszystkie"],
            index=1,
            key="rows_per_page"
        )
    
    # Obliczenia paginacji
    if rows_per_page == "Wszystkie":
        rows_per_page = total_rows
    else:
        rows_per_page = int(rows_per_page)
    
    total_pages = max(1, (total_rows + rows_per_page - 1) // rows_per_page)
    
    # Inicjalizacja strony
    if 'page' not in st.session_state:
        st.session_state.page = 1
    
    # Upewnij się, że numer strony jest w prawidłowym zakresie
    st.session_state.page = max(1, min(st.session_state.page, total_pages))
    
    # Wybór strony (tylko jeśli więcej niż 1 strona)
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            current_page = st.number_input(
                f"Strona (1-{total_pages})", 
                min_value=1, 
                max_value=total_pages, 
                value=st.session_state.page,
                key="page_selector"
            )
            st.session_state.page = current_page
    
    # Obliczenie zakresu wierszy
    start_idx = (st.session_state.page - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, total_rows)
    
    # Informacja o zakresie
    st.write(f"Wyświetlanie wierszy **{start_idx + 1}-{end_idx}** z **{total_rows}**")
    
    # Wyświetlenie danych
    safe_paginated_display(data_to_display, rows_per_page, st.session_state.page)

    # Zakładki analizy - USUNIĘTO GRUPOWANIE
    tab1, tab2, tab3 = st.tabs([
        "📈 Statystyki", 
        "🔧 Przetwarzanie", 
        "📊 Wizualizacje"
    ])

    # Zakładka 1: Statystyki (bez zmian)
    with tab1:
        st.header("📈 Analiza statystyczna")
        
        # Brakujące wartości
        st.subheader("❓ Brakujące wartości")
        display_missing_values(data)
        
        st.divider()
        
        # Statystyki numeryczne
        st.subheader("🔢 Statystyki dla kolumn numerycznych")
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_cols:
            num_stats = calculate_numerical_stats(data)
            if num_stats is not None:
                st.dataframe(num_stats, use_container_width=True)
                
                # Dodatkowe informacje
                with st.expander("ℹ️ Wyjaśnienie statystyk"):
                    st.markdown("""
                    - **count**: liczba niepustych wartości
                    - **mean**: średnia arytmetyczna
                    - **std**: odchylenie standardowe
                    - **min/max**: wartości minimalne/maksymalne
                    - **25%/50%/75%**: kwartyle (percentyle)
                    """)
        else:
            st.info("Brak kolumn numerycznych w zbiorze danych.")
        
        st.divider()
        
        # Statystyki kategoryczne
        st.subheader("🏷️ Statystyki dla kolumn kategorycznych")
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            cat_stats = calculate_categorical_stats(data)
            if cat_stats is not None:
                for col, stats in cat_stats.items():
                    with st.expander(f"**{col}**"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Unikalne wartości", stats['Unikalne wartości'])
                            st.metric("Brakujące wartości", stats['Brakujące wartości'])
                        with col2:
                            st.write(f"**Moda:** {stats['Moda']}")
                            st.write(f"**Liczba wystąpień mody:** {stats['Liczba wystąpień mody']}")
                        
                        st.write("**Top 5 wartości:**")
                        top_values = pd.DataFrame(
                            list(stats['Najczęstsze 5 wartości'].items()),
                            columns=['Wartość', 'Liczba wystąpień']
                        )
                        st.dataframe(top_values, use_container_width=True)
        else:
            st.info("Brak kolumn kategorycznych w zbiorze danych.")
        
        st.divider()
        
        # Korelacje
        st.subheader("🔗 Korelacje między atrybutami")
        
        if len(numeric_cols) >= 2:
            correlation_method = st.selectbox(
                "Wybierz metodę korelacji",
                ["pearson", "kendall", "spearman"],
                index=0,
                help="Pearson: liniowa, Kendall/Spearman: nieparametryczne"
            )

            corr_matrix = calculate_correlations(data, method=correlation_method)
            if corr_matrix is not None:
                # Wyświetl macierz korelacji
                st.dataframe(corr_matrix.round(3), use_container_width=True)
                
                # Opcjonalna mapa cieplna
                if st.checkbox("🌡️ Pokaż mapę cieplną korelacji"):
                    import plotly.express as px
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        title=f"Mapa cieplna korelacji ({correlation_method})",
                        color_continuous_scale='RdBu_r'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Nie można obliczyć korelacji - brak odpowiednich danych numerycznych.")
        else:
            st.info("Potrzebne są co najmniej 2 kolumny numeryczne do obliczenia korelacji.")

    # Zakładka 2: Przetwarzanie danych - PRZEPISANA
    with tab2:
        st.header("🔧 Przetwarzanie danych")

        # Informacje o obecnym stanie danych
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Obecny rozmiar", f"{data.shape[0]} × {data.shape[1]}")
        with col2:
            st.metric("Brakujące wartości", data.isnull().sum().sum())
        with col3:
            st.metric("Duplikaty", data.duplicated().sum())

        st.divider()

        # Wybór operacji
        processing_option = st.selectbox(
            "🎯 Wybierz operację przetwarzania",
            [
                "Operacje na wierszach", 
                "Usuwanie kolumn",
                "Zamiana wartości", 
                "Obsługa brakujących danych",
                "Usuwanie duplikatów", 
                "Skalowanie danych", 
                "Kodowanie zmiennych kategorycznych"
            ]
        )

        if processing_option == "Operacje na wierszach":
            st.subheader("🎯 Operacje na wierszach")
            
            operation_type = st.radio(
                "Typ operacji",
                ["Po indeksach", "Po wartościach w kolumnie"],
                key="row_operation_type"
            )
            
            if operation_type == "Po indeksach":
                st.write("**Operacje na wierszach według indeksów**")
                
                indices_help = """
                Wprowadź indeksy wierszy:
                - Pojedyncze: "1,3,5"
                - Zakresy: "1-5"  
                - Kombinacje: "1,3-5,7"
                """
                indices_str = st.text_input("Indeksy wierszy:", help=indices_help, key="row_indices")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("✅ Zachowaj wybrane wiersze", key="keep_rows_indices"):
                        if indices_str:
                            st.session_state.data = keep_rows_by_indices(st.session_state.data, indices_str)
                            st.rerun()
                        else:
                            st.warning("⚠️ Wprowadź indeksy wierszy")
                
                with col2:
                    if st.button("❌ Usuń wybrane wiersze", key="remove_rows_indices"):
                        if indices_str:
                            st.session_state.data = remove_rows_by_indices(st.session_state.data, indices_str)
                            st.rerun()
                        else:
                            st.warning("⚠️ Wprowadź indeksy wierszy")

            else:  # Po wartościach w kolumnie
                st.write("**Operacje na wierszach według wartości**")
                
                col = create_column_selector(data, "Wybierz kolumnę", multiselect=False, key="row_filter_col")
                if col:
                    # Pokaż unikalne wartości w kolumnie
                    unique_vals = data[col].value_counts().head(20)
                    with st.expander(f"🔍 Podgląd wartości w kolumnie '{col}'"):
                        st.dataframe(unique_vals, use_container_width=True)
                    
                    value = st.text_input("Podaj wartość:", key="row_filter_value")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("✅ Zachowaj wiersze z tą wartością", key="keep_rows_value"):
                            if value:
                                st.session_state.data = keep_rows_by_value(st.session_state.data, col, value)
                                st.rerun()
                            else:
                                st.warning("⚠️ Wprowadź wartość")
                    
                    with col2:
                        if st.button("❌ Usuń wiersze z tą wartością", key="remove_rows_value"):
                            if value:
                                st.session_state.data = remove_rows_by_value(st.session_state.data, col, value)
                                st.rerun()
                            else:
                                st.warning("⚠️ Wprowadź wartość")

        elif processing_option == "Usuwanie kolumn":
            st.subheader("❌ Usuwanie kolumn")
            
            cols_to_remove = create_column_selector(
                data, 
                "Wybierz kolumny do usunięcia",
                multiselect=True,
                key="cols_to_remove"
            )
            
            if cols_to_remove:
                st.warning(f"⚠️ Zostaną usunięte kolumny: {cols_to_remove}")
                
                if st.button("❌ Usuń wybrane kolumny", key="execute_remove_cols"):
                    st.session_state.data = remove_columns(st.session_state.data, cols_to_remove)
                    st.rerun()

        elif processing_option == "Zamiana wartości":
            st.subheader("🔄 Zamiana wartości w kolumnach")
            
            col = create_column_selector(data, "Wybierz kolumnę", multiselect=False, key="replace_col")
            
            if col:
                # Pokaż unikalne wartości
                unique_vals = data[col].value_counts().head(10)
                with st.expander(f"🔍 Aktualne wartości w kolumnie '{col}'"):
                    st.dataframe(unique_vals, use_container_width=True)
                
                old_value = st.text_input("Stara wartość", key="old_val")
                new_value = st.text_input("Nowa wartość", key="new_val")
                
                if st.button("🔄 Zamień wartości", key="execute_replace"):
                    if old_value != "":  # Pozwól na puste stringi
                        st.session_state.data = replace_values(st.session_state.data, col, old_value, new_value)
                        st.rerun()
                    else:
                        st.warning("⚠️ Wprowadź starą wartość")

        elif processing_option == "Obsługa brakujących danych":
            st.subheader("❓ Obsługa brakujących danych")

            # Pokaż obecne brakujące wartości
            missing_summary = data.isnull().sum()
            missing_cols = missing_summary[missing_summary > 0]
            
            if len(missing_cols) == 0:
                st.success("✅ Brak brakujących wartości w danych!")
            else:
                st.write("**Kolumny z brakującymi wartościami:**")
                missing_df = pd.DataFrame({
                    'Kolumna': missing_cols.index,
                    'Liczba brakujących': missing_cols.values,
                    'Procent': (missing_cols.values / len(data) * 100).round(1)
                })
                st.dataframe(missing_df, use_container_width=True)

                handling_method = st.radio(
                    "Wybierz metodę obsługi",
                    ["drop_rows", "drop_columns", "mean", "median", "mode", "zero"],
                    format_func=lambda x: {
                        "drop_rows": "Usuń wiersze z brakującymi wartościami",
                        "drop_columns": "Usuń kolumny z brakującymi wartościami",
                        "mean": "Wypełnij średnią (tylko numeryczne)",
                        "median": "Wypełnij medianą (tylko numeryczne)",
                        "mode": "Wypełnij modą (najczęstsza wartość)",
                        "zero": "Wypełnij zerem/pustym stringiem"
                    }[x],
                    key="missing_method"
                )

                target_columns = create_column_selector(
                    data[missing_cols.index], 
                    "Wybierz kolumny do przetworzenia (puste = wszystkie z brakującymi)",
                    multiselect=True,
                    key="missing_cols"
                )

                if st.button("🔧 Obsłuż brakujące wartości", key="execute_missing"):
                    columns_to_process = target_columns if target_columns else missing_cols.index.tolist()
                    st.session_state.data = handle_missing_values(
                        st.session_state.data,
                        method=handling_method,
                        columns=columns_to_process
                    )
                    st.rerun()

        elif processing_option == "Usuwanie duplikatów":
            st.subheader("🔄 Usuwanie duplikatów")

            dup_count = data.duplicated().sum()
            st.metric("Liczba duplikatów", dup_count)

            if dup_count > 0:
                # Pokaż przykłady duplikatów
                if st.checkbox("🔍 Pokaż przykłady duplikatów"):
                    duplicated_rows = data[data.duplicated(keep=False)].head(10)
                    st.dataframe(duplicated_rows, use_container_width=True)
                
                if st.button("❌ Usuń duplikaty", key="execute_remove_dups"):
                    st.session_state.data = remove_duplicates(st.session_state.data)
                    st.rerun()
            else:
                st.success("✅ Brak duplikatów w zbiorze danych")

        elif processing_option == "Skalowanie danych":
            st.subheader("📏 Skalowanie danych")

            # Tylko kolumny numeryczne
            num_cols = data.select_dtypes(include=['number']).columns.tolist()

            if not num_cols:
                st.warning("⚠️ Brak kolumn numerycznych do skalowania")
            else:
                # Pokaż zakresy wartości
                with st.expander("🔍 Obecne zakresy wartości"):
                    ranges_data = []
                    for col in num_cols:
                        ranges_data.append({
                            'Kolumna': col,
                            'Min': data[col].min(),
                            'Max': data[col].max(),
                            'Średnia': data[col].mean(),
                            'Odch. std.': data[col].std()
                        })
                    st.dataframe(pd.DataFrame(ranges_data), use_container_width=True)

                scale_method = st.radio(
                    "Metoda skalowania",
                    ["minmax", "standard"],
                    format_func=lambda x: {
                        "minmax": "Min-Max (0-1)", 
                        "standard": "Standardyzacja (średnia=0, odch.std=1)"
                    }[x],
                    key="scale_method"
                )

                cols_to_scale = create_column_selector(
                    data[num_cols], 
                    "Wybierz kolumny do przeskalowania",
                    multiselect=True,
                    key="scale_cols"
                )

                if st.button("📏 Skaluj dane", key="execute_scale"):
                    if cols_to_scale:
                        st.session_state.data = scale_data(st.session_state.data, cols_to_scale, method=scale_method)
                        st.rerun()
                    else:
                        st.warning("⚠️ Wybierz kolumny do skalowania")

        elif processing_option == "Kodowanie zmiennych kategorycznych":
            st.subheader("🏷️ Kodowanie zmiennych kategorycznych")

            # Tylko kolumny kategoryczne
            cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

            if not cat_cols:
                st.warning("⚠️ Brak kolumn kategorycznych do kodowania")
            else:
                # Pokaż informacje o kolumnach kategorycznych
                with st.expander("🔍 Informacje o kolumnach kategorycznych"):
                    cat_info = []
                    for col in cat_cols:
                        cat_info.append({
                            'Kolumna': col,
                            'Unikalne wartości': data[col].nunique(),
                            'Najczęstsza': data[col].mode().iloc[0] if not data[col].mode().empty else 'N/A',
                            'Brakujące': data[col].isnull().sum()
                        })
                    st.dataframe(pd.DataFrame(cat_info), use_container_width=True)

                encoding_method = st.radio(
                    "Metoda kodowania",
                    ["onehot", "binary"],
                    format_func=lambda x: {
                        "onehot": "One-Hot Encoding (każda kategoria = osobna kolumna)", 
                        "binary": "Binary Encoding (reprezentacja binarna)"
                    }[x],
                    key="encoding_method"
                )

                cols_to_encode = create_column_selector(
                    data[cat_cols], 
                    "Wybierz kolumny do zakodowania",
                    multiselect=True,
                    key="encode_cols"
                )

                if st.button("🏷️ Koduj dane", key="execute_encode"):
                    if cols_to_encode:
                        st.session_state.data = encode_categorical(st.session_state.data, cols_to_encode, method=encoding_method)
                        st.rerun()
                    else:
                        st.warning("⚠️ Wybierz kolumny do kodowania")

    # Zakładka 3: Wizualizacja (bez zmian)
    with tab3:
        st.header("Wizualizacja danych")

        viz_type = st.selectbox(
            "Wybierz typ wykresu",
            ["Histogram", "Wykres pudełkowy", "Wykres punktowy",
             "Wykres słupkowy", "Wykres kołowy", "Wykres par"]
        )

        if viz_type == "Histogram":
            st.subheader("Histogram")
            
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

else:
    # Gdy brak danych
    st.title("📊 Analizator danych CSV")
    
    st.markdown("""
    ## Witaj w analizatorze danych! 👋
    
    Ta aplikacja pozwala na:
    - 📁 **Wczytywanie** plików CSV
    - 📈 **Analizę statystyczną** danych
    - 🔧 **Przetwarzanie** i czyszczenie danych  
    - 📊 **Wizualizację** wyników
    
    ### Jak zacząć?
    1. Wczytaj plik CSV używając panelu po lewej stronie ⬅️
    2. Sprawdź jakość danych w zakładce "Statystyki"
    3. Ewentualnie oczyść dane w zakładce "Przetwarzanie"
    4. Twórz wizualizacje i analizuj!
    
    ### Obsługiwane pliki CSV
    - Różne separatory (`,` `;` `|` tab)
    - Różne kodowania (UTF-8, Latin-1, itp.)
    - Automatyczne wykrywanie brakujących wartości
    """)
    
    # Przykład struktury CSV
    st.subheader("📋 Przykład poprawnego pliku CSV")
    
    example_csv = """nazwa,wiek,miasto,zarobki
Jan Kowalski,25,Warszawa,5000
Anna Nowak,30,Kraków,6000
Piotr Wiśniewski,35,Gdańsk,5500"""
    
    st.code(example_csv, language="csv")
    
    st.info("💡 **Wskazówka:** Pierwszy wiersz powinien zawierać nazwy kolumn (nagłówki)")