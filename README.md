# Aplikacja do analizy zbioru Credit Approval

Aplikacja stworzona w Streamlit do kompleksowej analizy zbioru danych "Credit Approval" z UCI Machine Learning Repository.

## Opis

Aplikacja umożliwia analizę zbioru danych Credit Approval, który zawiera informacje o wnioskach kredytowych. Umożliwia wczytanie danych, przetwarzanie, wizualizację oraz analizę modelową.

## Funkcjonalności

### Wczytywanie danych
- Wczytywanie danych z pliku CSV
- Wczytywanie przykładowego zbioru "Credit Approval"

### Statystyki
- Obliczanie i wyświetlanie miar statystycznych dla cech numerycznych (min, max, średnia, odchylenie standardowe, mediana, skośność, kurtoza)
- Obliczanie statystyk dla cech kategorycznych (unikalne wartości, moda, liczba wystąpień mody)
- Wyznaczenie korelacji między atrybutami (metody: Pearson, Kendall, Spearman)

### Przetwarzanie danych
- Usuwanie wybranych wierszy lub kolumn po numerach lub nazwach
- Zamiana wartości w dowolnej kolumnie (ręcznie lub automatycznie)
- Obsługa brakujących danych (usuwanie lub wypełnianie)
- Usuwanie duplikatów wierszy
- Skalowanie i standaryzacja wybranych kolumn (MinMaxScaler, StandardScaler)
- Kodowanie kolumn symbolicznych (One-Hot Encoding, Binary Encoding)

### Wizualizacja
- Histogramy
- Wykresy pudełkowe
- Wykresy punktowe
- Wykresy słupkowe
- Wykresy kołowe
- Mapy cieplne korelacji
- Wykresy par

### Modelowanie
- Klasyfikacja (logistyczna, drzewa decyzyjne, lasy losowe, SVM, k-NN)
- Grupowanie K-Means (z doborem liczby klastrów)
- Reguły asocjacyjne Apriori (z parametrami podpory i pewności)

## Struktura projektu

```
credit_approval_app/
├── app.py                  # Główna aplikacja Streamlit
├── requirements.txt        # Zależności projektu
├── modules/
│   ├── __init__.py         # Inicjalizacja pakietu
│   ├── data_loader.py      # Wczytywanie danych
│   ├── statistics.py       # Analiza statystyczna
│   ├── data_processing.py  # Przetwarzanie danych
│   ├── visualization.py    # Wizualizacja danych
│   └── modeling.py         # Modele uczenia maszynowego
└── README.md               # Dokumentacja projektu
```

## Instalacja i uruchomienie


1. Stworz wirtualne srodowisko:
```
python -m venv .venv
```

2. Aktywuj venva:
```
.venv\Scripts\activate.bat
```

3. Zainstaluj wymagane pakiety:
```
pip install -r requirements.txt
```

4. Uruchom aplikację:
```
python main.py
```

## Zbiór danych

Zbiór "Credit Approval" pochodzi z UCI Machine Learning Repository i zawiera dane dotyczące wniosków o wydanie karty kredytowej. Ze względów prywatności, nazwy atrybutów (A1-A16) oraz wartości są zaszyfrowane.

Źródło: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/27/credit+approval)