import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def concat_datasets(train, test, group_cols, date_col='date'):
    """
        Concatena los conjuntos de datos de entrenamiento y prueba para ingeniería de características de series temporales.
        
        Esta función combina los conjuntos de entrenamiento y prueba en un único DataFrame, agregando
        una columna 'split' para identificar cada subconjunto. Los datos se ordenan por tienda, artículo
        y fecha para asegurar el ordenamiento adecuado para los cálculos de rezago/promedios móviles.
        
        Parámetros
        ----------
        train : pd.DataFrame
            Conjunto de datos de entrenamiento que contiene al menos las columnas 'store', 'item' y 'date'
        test : pd.DataFrame
            Conjunto de datos de prueba con la misma estructura que train
        group_cols : list
            Columnas que definen series temporales independientes
        date_col : str, default='date'
            Nombre de la columna de fecha/timestamp
        
        Retorna
        -------
        pd.DataFrame
            Conjunto de datos combinado y ordenado
    """
    tr = train.copy(); te = test.copy()
    tr['split'] = 'train'; te['split'] = 'test'
    
    df = pd.concat([tr, te], ignore_index=True)
    sort_cols = group_cols + [date_col]
    df = df.sort_values(sort_cols)
    df.reset_index(drop=True, inplace=True)
    
    print(f"✅ Datasets concatenated and sorted by {sort_cols}. Total rows: {len(df):,}")
    return df

def set_calendar_features(df):
    """
        Crea características basadas en el calendario a partir de la columna 'date'.
        
        Extrae características temporales que capturan patrones estacionales, ciclos semanales
        y comportamientos específicos del mes. Todas las características están optimizadas en
        memoria usando tipos enteros apropiados.
        
        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame con una columna 'date' (debe ser de tipo datetime)
        
        Retorna
        -------
        None
            Modifica df in-place agregando las siguientes columnas:
            - day_of_week (int8): 0=Lunes, 6=Domingo
            - is_weekend (int8): 1 si es Sábado/Domingo, 0 en caso contrario
            - month (int8): 1-12
            - year (int16): Año completo
            - day (int8): Día del mes (1-31)
            - week_of_year (int16): Número de semana ISO (1-53)
            - quarter (int8): Trimestre del año (1-4)
            - month_start (int8): 1 si es el primer día del mes, 0 en caso contrario
            - month_end (int8): 1 si es el último día del mes, 0 en caso contrario
    """
    dt = df['date'].dt
    df['day_of_week'] = dt.dayofweek.astype('int8')
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype('int8')
    df['month'] = dt.month.astype('int8')
    df['year'] = dt.year.astype('int16')
    df['day'] = dt.day.astype('int8')
    df['week_of_year'] = dt.isocalendar().week.astype('int16')
    df['quarter'] = dt.quarter.astype('int8')
    df['month_start'] = dt.is_month_start.astype('int8')
    df['month_end'] = dt.is_month_end.astype('int8')
    
    print("✅ Calendar features created: day_of_week, is_weekend, month, year, day, week_of_year, quarter, month_start, month_end")
    
def set_lags_and_rolling(df, group_cols, target_col, lags=[1, 7, 14, 28], rolling_windows=[7, 28]):
    """
        Crea características de rezago y promedios móviles para pronóstico de series temporales.
        
        Las características de rezago capturan los valores históricos en puntos temporales específicos,
        mientras que los promedios móviles suavizan las fluctuaciones de corto plazo. Ambos se calculan
        por grupo para mantener la independencia entre diferentes series temporales.
        
        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame ordenado por fecha dentro de cada grupo
        group_cols : list of str
            Columnas que definen series temporales independientes (ej., ['tienda', 'artículo'])
        target_col : str
            Nombre de la variable objetivo para crear rezagos (ej., 'ventas')
        lags : list of int, default=[1, 7, 14, 28]
            Lista de períodos de rezago a crear. Opciones comunes:
            - 1: valor de ayer (inercia)
            - 7: mismo día de la semana pasada (patrón semanal)
            - 14: hace dos semanas
            - 28: aproximadamente el mismo día del mes pasado
        rolling_windows : list of int, default=[7, 28]
            Lista de tamaños de ventana para promedios móviles (ej., 7 para promedio semanal)
        
        Retorna
        -------
        None
            Modifica df in-place agregando:
            - columnas lag_{L} para cada rezago en lags
            - columnas roll{W} para cada ventana en rolling_windows
    """
    gr = df.groupby(group_cols, group_keys=False)
    for L in lags:
        df[f'lag_{L}'] = gr[target_col].shift(L)
    for W in rolling_windows:
        df[f'roll{W}'] = gr[target_col].shift(1).rolling(W, min_periods=1).mean()
    
    lag_features = [f'lag_{L}' for L in lags]
    roll_features = [f'roll{W}' for W in rolling_windows]
    print(f"✅ Lag features created: {lag_features}")
    print(f"✅ Rolling features created: {roll_features}")
        
def to_categorical(df, cols):
    """
        Convierte las columnas especificadas al tipo categorical para eficiencia de memoria.
        
        La codificación categórica es beneficiosa para modelos basados en árboles (especialmente LightGBM)
        ya que reduce el uso de memoria y habilita algoritmos de división optimizados.
        
        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame que contiene las columnas a convertir
        cols : list of str
            Lista de nombres de columnas a convertir a categorical
        
        Retorna
        -------
        None
            Modifica df in-place convirtiendo las columnas especificadas al tipo categorical
    """
    for cat in cols:
        df[cat] = df[cat].astype('int16').astype('category')
    print(f"✅ Converted to categorical: {cols}")
        
def downcast(df, num_cols):
    """
        Reduce el tamaño de las columnas numéricas a tipos de datos más pequeños para reducir el uso de memoria.
        
        Convierte float64 a float32 (o menor si es posible) con pérdida mínima de precisión,
        reduciendo significativamente la huella de memoria para conjuntos de datos grandes.
        
        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame que contiene las columnas numéricas
        num_cols : list of str
            Lista de nombres de columnas numéricas a reducir
        
        Retorna
        -------
        None
            Modifica df in-place reduciendo el tamaño de las columnas numéricas especificadas
    """
    df[num_cols] = df[num_cols].apply(pd.to_numeric, downcast='float')
    print(f"✅ Downcasted {len(num_cols)} numeric columns to float32")
    
def split(df, cutoff, feat_vars, target_col):
    """
        Realiza la división temporal de entrenamiento/validación/prueba para pronóstico de series temporales.
        
        Divide los datos temporalmente (no aleatoriamente) para respetar la naturaleza causal de
        las series temporales. Los datos de entrenamiento vienen antes de la fecha de corte, los datos
        de validación vienen después, y los datos de prueba son el conjunto de prueba original.
        
        Parámetros
        ----------
        df : pd.DataFrame
            Conjunto de datos combinado con la columna 'split' creada por concat_datasets()
        cutoff : pd.Timestamp
            Fecha que separa los conjuntos de entrenamiento y validación (ej., '2017-10-01')
        feat_vars : list of str
            Lista de nombres de columnas de características a usar para el modelado
        target_col : str
            Nombre de la columna de la variable objetivo
        
        Retorna
        -------
        X_train : pd.DataFrame
            Características de entrenamiento (date < cutoff)
        y_train : pd.Series
            Objetivo de entrenamiento (float32)
        X_val : pd.DataFrame
            Características de validación (date >= cutoff)
        y_val : pd.Series
            Objetivo de validación (float32)
        X_test : pd.DataFrame
            Características de prueba (conjunto de prueba original)
    """
    tr_df = df[df['split'] == 'train'].copy()
    te_df = df[df['split'] == 'test'].copy()

    tr_mask = tr_df['date'] < cutoff
    val_mask = tr_df['date'] >= cutoff

    X_train = tr_df.loc[tr_mask, feat_vars]
    y_train = tr_df.loc[tr_mask, target_col].astype('float32')
    X_val = tr_df.loc[val_mask, feat_vars]
    y_val = tr_df.loc[val_mask, target_col].astype('float32')
    X_test = te_df[feat_vars]

    print(f'X_train shape: {X_train.shape}\nX_val shape: {X_val.shape}\nX_test shape: {X_test.shape}')
    return X_train, y_train, X_val, y_val, X_test

