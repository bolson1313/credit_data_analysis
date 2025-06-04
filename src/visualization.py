import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def histogram(data, column, bins=20, title=None):
    """Tworzy histogram dla kolumny numerycznej."""
    if data is None or column not in data.columns:
        return None

    if not pd.api.types.is_numeric_dtype(data[column]):
        return None

    if title is None:
        title = f"Histogram dla {column}"
    
    fig = px.histogram(
        data, 
        x=column, 
        nbins=bins, 
        title=title,
        labels={column: column}
    )
    
    # Dodaj statystyki do tytułu
    mean_val = data[column].mean()
    std_val = data[column].std()
    fig.add_annotation(
        text=f"Średnia: {mean_val:.2f}, Odch. std: {std_val:.2f}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    return fig


def box_plot(data, column, title=None):
    """Tworzy wykres pudełkowy dla kolumny numerycznej."""
    if data is None or column not in data.columns:
        return None

    if not pd.api.types.is_numeric_dtype(data[column]):
        return None

    if title is None:
        title = f"Wykres pudełkowy dla {column}"
    
    fig = px.box(
        data, 
        y=column, 
        title=title,
        labels={column: column}
    )
    
    # Dodaj statystyki
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    median = data[column].median()
    
    fig.add_annotation(
        text=f"Q1: {q1:.2f}, Mediana: {median:.2f}, Q3: {q3:.2f}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    return fig


def scatter_plot(data, x_column, y_column, color_column=None, title=None):
    """Tworzy wykres punktowy dla dwóch kolumn numerycznych."""
    if data is None or x_column not in data.columns or y_column not in data.columns:
        return None

    if not pd.api.types.is_numeric_dtype(data[x_column]) or not pd.api.types.is_numeric_dtype(data[y_column]):
        return None

    if color_column and color_column not in data.columns:
        color_column = None

    if title is None:
        title = f"{x_column} vs {y_column}"
    
    fig = px.scatter(
        data, 
        x=x_column, 
        y=y_column, 
        color=color_column, 
        title=title,
        labels={x_column: x_column, y_column: y_column}
    )
    
    # Dodaj linię trendu jeśli nie ma kolorowania
    if color_column is None:
        # Oblicz korelację
        correlation = data[x_column].corr(data[y_column])
        
        # Dodaj linię trendu
        z = np.polyfit(data[x_column].dropna(), data[y_column].dropna(), 1)
        p = np.poly1d(z)
        
        x_trend = np.linspace(data[x_column].min(), data[x_column].max(), 100)
        y_trend = p(x_trend)
        
        fig.add_trace(go.Scatter(
            x=x_trend, 
            y=y_trend,
            mode='lines',
            name=f'Trend (r={correlation:.3f})',
            line=dict(color='red', dash='dash')
        ))
    
    return fig


def bar_chart(data, x_column, y_column=None, title=None):
    """Tworzy wykres słupkowy."""
    if data is None or x_column not in data.columns:
        return None

    if y_column and y_column not in data.columns:
        y_column = None

    if y_column:
        # Wykres wartości
        if title is None:
            title = f'{y_column} według {x_column}'
        
        # Grupuj i agreguj dane
        grouped_data = data.groupby(x_column)[y_column].mean().reset_index()
        
        fig = px.bar(
            grouped_data, 
            x=x_column, 
            y=y_column, 
            title=title,
            labels={x_column: x_column, y_column: f'Średnia {y_column}'}
        )
    else:
        # Wykres liczności
        counts = data[x_column].value_counts().reset_index()
        counts.columns = [x_column, 'liczność']
        
        if title is None:
            title = f'Liczność wartości w kolumnie {x_column}'
        
        fig = px.bar(
            counts, 
            x=x_column, 
            y='liczność', 
            title=title,
            labels={x_column: x_column, 'liczność': 'Liczność'}
        )

    return fig


def pie_chart(data, column, title=None, max_categories=10):
    """Tworzy wykres kołowy dla kolumny kategorycznej."""
    if data is None or column not in data.columns:
        return None

    # Policz wartości
    counts = data[column].value_counts()
    
    # Ogranicz liczbę kategorii
    if len(counts) > max_categories:
        top_counts = counts.head(max_categories - 1)
        others_count = counts.tail(len(counts) - max_categories + 1).sum()
        
        # Dodaj kategorię "Inne"
        top_counts['Inne'] = others_count
        counts = top_counts

    counts_df = counts.reset_index()
    counts_df.columns = [column, 'liczność']

    if title is None:
        title = f'Rozkład wartości w kolumnie {column}'

    fig = px.pie(
        counts_df, 
        values='liczność', 
        names=column, 
        title=title
    )
    
    # Dodaj procenty
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig


def pair_plot(data, columns=None, hue=None, title=None):
    """Tworzy wykres par dla wybranych kolumn numerycznych."""
    if data is None:
        return None

    if columns is None:
        # Użyj wszystkich kolumn numerycznych, jeśli nie określono
        columns = data.select_dtypes(include=['number']).columns.tolist()
    else:
        # Filtruj, aby uwzględnić tylko kolumny numeryczne
        columns = [col for col in columns if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]

    if len(columns) < 2:
        return None

    if hue and hue not in data.columns:
        hue = None

    # Ograniczenie do maksymalnie 5 kolumn dla czytelności
    if len(columns) > 5:
        columns = columns[:5]
        st.warning(f"Ograniczono do pierwszych 5 kolumn: {columns}")

    # Stwórz siatkę wykresów punktowych plotly
    n = len(columns)
    fig = make_subplots(
        rows=n, cols=n,
        subplot_titles=[f"{col1} vs {col2}" if i != j else f"Rozkład {col1}" 
                       for i, col1 in enumerate(columns) 
                       for j, col2 in enumerate(columns)],
        vertical_spacing=0.05,
        horizontal_spacing=0.05
    )

    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i == j:  # Przekątna: histogramy
                fig.add_trace(
                    go.Histogram(
                        x=data[col1], 
                        name=col1,
                        showlegend=False,
                        nbinsx=20
                    ),
                    row=i + 1, col=j + 1
                )
            else:  # Poza przekątną: wykresy punktowe
                if hue:
                    # Kolorowanie według zmiennej hue
                    unique_hues = data[hue].unique()
                    for k, hue_val in enumerate(unique_hues):
                        subset = data[data[hue] == hue_val]
                        fig.add_trace(
                            go.Scatter(
                                x=subset[col2], 
                                y=subset[col1],
                                mode='markers',
                                name=f"{hue}={hue_val}",
                                showlegend=(i == 0 and j == 1),  # Legenda tylko raz
                                marker=dict(size=4)
                            ),
                            row=i + 1, col=j + 1
                        )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=data[col2], 
                            y=data[col1], 
                            mode='markers', 
                            showlegend=False,
                            marker=dict(size=4)
                        ),
                        row=i + 1, col=j + 1
                    )

    # Aktualizacja układu
    fig.update_layout(
        height=150 * n,
        title=title or "Wykres par",
        showlegend=bool(hue)
    )

    return fig


def correlation_heatmap(data, title=None, method='pearson'):
    """Tworzy mapę cieplną korelacji."""
    if data is None:
        return None
    
    # Tylko kolumny numeryczne
    numeric_data = data.select_dtypes(include=['number'])
    
    if numeric_data.shape[1] < 2:
        return None
    
    # Oblicz korelacje
    corr_matrix = numeric_data.corr(method=method)
    
    if title is None:
        title = f'Mapa cieplna korelacji ({method})'
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title=title,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1
    )
    
    return fig


def distribution_comparison(data, column, group_column, title=None):
    """Porównuje rozkłady wartości między grupami."""
    if data is None or column not in data.columns or group_column not in data.columns:
        return None
    
    if not pd.api.types.is_numeric_dtype(data[column]):
        return None
    
    if title is None:
        title = f'Porównanie rozkładu {column} według {group_column}'
    
    fig = px.violin(
        data,
        x=group_column,
        y=column,
        box=True,
        title=title,
        labels={column: column, group_column: group_column}
    )
    
    return fig