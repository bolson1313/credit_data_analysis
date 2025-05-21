import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .data_loader import get_display_column_name


def histogram(data, column, bins=20, title=None):
    """Tworzy histogram dla kolumny numerycznej."""
    if data is None or column not in data.columns:
        return None

    if not pd.api.types.is_numeric_dtype(data[column]):
        return None

    display_name = get_display_column_name(column)
    if title is None:
        title = f"Histogram dla {display_name}"
    fig = px.histogram(data, x=column, nbins=bins, title=title)
    return fig


def box_plot(data, column, title=None):
    """Tworzy wykres pudełkowy dla kolumny numerycznej."""
    if data is None or column not in data.columns:
        return None

    if not pd.api.types.is_numeric_dtype(data[column]):
        return None

    display_name = get_display_column_name(column)
    if title is None:
        title = f"Wykres pudełkowy dla {display_name}"
    fig = px.box(data, y=column, title=title)
    return fig


def scatter_plot(data, x_column, y_column, color_column=None, title=None):
    """Tworzy wykres punktowy dla dwóch kolumn numerycznych."""
    if data is None or x_column not in data.columns or y_column not in data.columns:
        return None

    if not pd.api.types.is_numeric_dtype(data[x_column]) or not pd.api.types.is_numeric_dtype(data[y_column]):
        return None

    if color_column and color_column not in data.columns:
        color_column = None

    x_display = get_display_column_name(x_column)
    y_display = get_display_column_name(y_column)
    if title is None:
        title = f"{x_display} vs {y_display}"
    fig = px.scatter(data, x=x_column, y=y_column, color=color_column, title=title)
    return fig


def bar_chart(data, x_column, y_column=None, title=None):
    """Tworzy wykres słupkowy."""
    if data is None or x_column not in data.columns:
        return None

    if y_column and y_column not in data.columns:
        y_column = None

    x_display = get_display_column_name(x_column)
    if y_column:
        y_display = get_display_column_name(y_column)
        fig = px.bar(data, x=x_column, y=y_column, title=title or f'Wykres słupkowy: {y_display} według {x_display}')
    else:
        # Wykres liczności
        counts = data[x_column].value_counts().reset_index()
        counts.columns = [x_column, 'liczność']
        fig = px.bar(counts, x=x_column, y='liczność', title=title or f'Liczność wartości w kolumnie {x_display}')

    return fig


def pie_chart(data, column, title=None):
    """Tworzy wykres kołowy dla kolumny kategorycznej."""
    if data is None or column not in data.columns:
        return None

    display_name = get_display_column_name(column)
    # Policz wartości
    counts = data[column].value_counts().reset_index()
    counts.columns = [column, 'liczność']

    fig = px.pie(counts, values='liczność', names=column, title=title or f'Rozkład wartości w kolumnie {display_name}')
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

    # Stwórz siatkę wykresów punktowych plotly
    n = len(columns)
    fig = make_subplots(rows=n, cols=n,
                        subplot_titles=[f"{get_display_column_name(col1)} vs {get_display_column_name(col2)}" for col1 in columns for col2 in columns])

    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i == j:  # Przekątna: histogramy
                fig.add_trace(
                    go.Histogram(x=data[col1], name=get_display_column_name(col1)),
                    row=i + 1, col=j + 1
                )
            else:  # Poza przekątną: wykresy punktowe
                if hue:
                    for hue_val in data[hue].unique():
                        subset = data[data[hue] == hue_val]
                        fig.add_trace(
                            go.Scatter(x=subset[col2], y=subset[col1],
                                       mode='markers',
                                       name=f"{get_display_column_name(hue)}={hue_val}",
                                       showlegend=(i == 0 and j == 1)),  # Wyświetlenie legendy jednokrotnie
                            row=i + 1, col=j + 1
                        )
                else:
                    fig.add_trace(
                        go.Scatter(x=data[col2], y=data[col1], mode='markers', showlegend=False),
                        row=i + 1, col=j + 1
                    )

    # Aktualizacja układu
    fig.update_layout(
        height=200 * n,
        width=200 * n,
        title=title or "Wykres par"
    )

    return fig