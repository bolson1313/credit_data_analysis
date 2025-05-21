import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
        if st.button("Wczytaj przykładowy zbiór Credit Approval"):
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
    
    # Kontrolka do wyboru liczby wierszy na stronie
    rows_per_page = st.selectbox(
        "Liczba wierszy na stronie",
        options=[10, 20, 50, 100, 500, "Wszystkie"],
        index=0
    )
    
    # Obliczenie całkowitej liczby stron
    total_rows = len(data)
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
    
    # Wyświetlenie danych
    safe_display_dataframe(data.iloc[start_idx:end_idx])

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
        
        # Wybór kolumn do grupowania
        st.subheader("Wybór danych")
        clustering_cols = create_column_selector(
            data,
            "Wybierz kolumny do grupowania (tylko numeryczne będą użyte)",
            multiselect=True,
            key="clustering_columns"
        )

        # Parametry grupowania
        st.subheader("Parametry grupowania")
        n_clusters = st.slider(
            "Liczba klastrów (k)", 
            min_value=2, 
            max_value=10, 
            value=3
        )

        # Przycisk do uruchomienia grupowania
        if st.button("Wykonaj grupowanie"):
            with st.spinner("Grupowanie danych..."):
                # Przygotowanie danych
                X_scaled, X_original = prepare_data_for_clustering(data, clustering_cols)

                if X_scaled is None:
                    st.error("Błąd przygotowania danych. Sprawdź, czy wybrane kolumny są odpowiednie.")
                else:
                    try:
                        # Trenowanie modelu
                        model = train_kmeans_model(X_scaled, n_clusters)

                        # Ewaluacja
                        metrics = evaluate_clustering(X_scaled, model)

                        if metrics is not None:
                            st.success("Grupowanie zakończone pomyślnie!")

                            # Wyświetl metryki
                            st.subheader("Metryki grupowania")
                            st.write(f"Inertia (suma kwadratów odległości): {metrics['inertia']:.2f}")
                            st.write(f"Współczynnik sylwetki: {metrics['silhouette']:.3f}")

                            # Rozmiary klastrów
                            st.subheader("Rozmiary klastrów")
                            sizes_df = pd.DataFrame.from_dict(
                                metrics['cluster_sizes'], 
                                orient='index', 
                                columns=['Liczba próbek']
                            )
                            safe_display_dataframe(sizes_df)

                            # Wizualizacja 2D (jeśli wybrano co najmniej 2 kolumny)
                            if len(X_scaled.columns) >= 2:
                                st.subheader("Wizualizacja klastrów")
                                
                                # Wybór cech do wizualizacji
                                viz_cols = st.multiselect(
                                    "Wybierz 2 cechy do wizualizacji",
                                    options=X_scaled.columns,
                                    default=list(X_scaled.columns[:2])
                                )

                                if len(viz_cols) == 2:
                                    fig = plot_clusters_2d(
                                        X_scaled, 
                                        metrics['labels'],
                                        metrics['centroids'],
                                        features=viz_cols
                                    )
                                    st.plotly_chart(fig, use_container_width=True, key="clusters_plot")
                                else:
                                    st.warning("Wybierz dokładnie 2 cechy do wizualizacji.")

                            # Dodaj etykiety klastrów do oryginalnych danych
                            clustered_data = X_original.copy()
                            clustered_data['Klaster'] = metrics['labels']
                            st.subheader("Dane z przypisanymi klastrami")
                            safe_display_dataframe(clustered_data)

                    except Exception as e:
                        st.error(f"Wystąpił błąd podczas grupowania: {str(e)}")
else:
    st.info("Wczytaj dane, aby rozpocząć analizę.")
    st.write("Wybierz opcję wczytywania danych w panelu bocznym.")