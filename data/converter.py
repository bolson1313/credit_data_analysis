import pandas as pd

# Wczytaj plik .data
data = pd.read_csv('data\crx.data', header=None)  # często pliki .data nie mają nagłówków

# Zapisz jako CSV
data.to_csv('plik.csv', index=False)