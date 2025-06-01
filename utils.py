import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def load_data(url: str) -> pl.DataFrame:
    """
    Carga un archivo CSV desde una URL.

    Args:
        url (str): URL del archivo CSV.

    Returns:
        pl.DataFrame: DataFrame con los datos cargados.
    """
    return pl.read_csv(url)


def remove_outliers(df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
    """
    Elimina outliers univariados usando el rango intercuartílico (IQR).

    Args:
        df (pl.DataFrame): DataFrame original.
        columns (List[str]): Lista de columnas a evaluar.

    Returns:
        pl.DataFrame: DataFrame sin outliers.
    """
    for col in columns:
        if df[col].dtype in [pl.Float64, pl.Int64]:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df = df.filter((pl.col(col) >= lower) & (pl.col(col) <= upper))
    return df


def scale_columns(df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
    """
    Aplica escalamiento estándar (Z-score) a las columnas numéricas.

    Args:
        df (pl.DataFrame): DataFrame original.
        columns (List[str]): Columnas a escalar.

    Returns:
        pl.DataFrame: DataFrame con columnas escaladas.
    """
    scaler = StandardScaler()
    scaled = [pl.Series(f"{col}_scaled", scaler.fit_transform(df[col].to_numpy().reshape(-1, 1)).flatten()) for col in columns]
    return df.with_columns(scaled)


def evaluate_model(X: np.ndarray, y: np.ndarray) -> float:
    """
    Evalúa un modelo de regresión lineal con train/test split (80/20).

    Args:
        X (np.ndarray): Matriz de características.
        y (np.ndarray): Vector objetivo.

    Returns:
        float: R² score del modelo.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)


def train_final_model(X: np.ndarray, y: np.ndarray) -> Tuple[LinearRegression, float]:
    """
    Entrena y evalúa un modelo final de regresión lineal.

    Args:
        X (np.ndarray): Características seleccionadas.
        y (np.ndarray): Objetivo.

    Returns:
        Tuple[LinearRegression, float]: Modelo entrenado y R² en test.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, r2_score(y_test, y_pred)


def plot_evolution(logbook) -> None:
    """
    Grafica la evolución del coeficiente R² por generación.

    Args:
        logbook: Registro generado por DEAP.
    """
    import matplotlib.pyplot as plt
    gens = logbook.select('gen')
    maxs = logbook.select('max')
    avgs = logbook.select('avg')

    plt.figure(figsize=(10, 6))
    plt.plot(gens, maxs, label="Máximo R²")
    plt.plot(gens, avgs, label="Promedio R²")
    plt.xlabel("Generación")
    plt.ylabel("R²")
    plt.title("Evolución del R²")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Grafica las predicciones frente a los valores reales.

    Args:
        y_true (np.ndarray): Valores reales.
        y_pred (np.ndarray): Valores predichos.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
    plt.xlabel('Valores Reales (escalados)')
    plt.ylabel('Predicciones (escalados)')
    plt.title('Predicciones vs Valores Reales')
    plt.grid(True)
    plt.show()
