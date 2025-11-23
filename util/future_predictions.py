"""
PREDICCIONES FUTURAS PARA DASHBOARD
===========================================

Genera predicciones para los próximos N días y crea archivo para Looker Studio.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from util import forecasting_fe

def generate_future_predictions(train, test, model, group_cols, target_col, n_days_ahead=30):
    # 1. Combinar todo el histórico posible
    if test is None and target_col in test.columns:
        historical = pd.concat([train, test], ignore_index=True)
        print('Using train + test')
    else:
        historical = train.copy()
        print('Using only train')
        
    historical = historical[group_cols + ['date', target_col]].copy()
    last_date = historical['date'].max()
    print(f'Last date: {last_date}')
    print(f'Historical: {len(historical):,} rows')
    
    # 2. Crear fechas futuras
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=n_days_ahead,
        freq='D'
    )
    print(f'Prediction from {future_dates[0]} to {future_dates[-1]}')

    # 3. Crear combinaciones store-item para cada fecha futura
    unique_comb = historical[group_cols].drop_duplicates()
    
    future_df = pd.DataFrame()
    for date in future_dates:
        temp = unique_comb.copy()
        temp['date'] = date
        future_df = pd.concat([future_df, temp], ignore_index=True)
    print(f'Total predictions to make: {len(future_df):,}')
    
    # 4. Combinar historico + futuro
    combined_df = pd.concat([
        historical,
        future_df.assign(**{target_col: np.nan})
    ], ignore_index=True)
    
    combined_df = combined_df.sort_values(group_cols + ['date']).reset_index(drop=True)
    
    # 5. Crear features
    forecasting_fe.set_calendar_features(combined_df)
    
    forecasting_fe.set_lags_and_rolling(
        df=combined_df,
        group_cols=group_cols,
        target_col=target_col,
        lags=[1, 7, 14, 28],
        rolling_windows=[7, 28]
    )
    
    # 6. Filtrar solo fechas futuras
    future_with_features = combined_df[combined_df['date'].isin(future_dates)].copy()
    
    # 7. Preparar para el modelo
    forecasting_fe.to_categorical(future_with_features, cols=group_cols)
    num_cols = ['lag_1', 'lag_7', 'lag_14', 'lag_28', 'roll7', 'roll28']
    forecasting_fe.downcast(future_with_features, num_cols=num_cols)
    
    feature_cols = [
        'store', 'item', 'day_of_week', 'is_weekend', 'month', 'year',
        'day', 'week_of_year', 'quarter', 'month_start', 'month_end',
        'lag_1', 'lag_7', 'lag_14', 'lag_28', 'roll7', 'roll28'
    ]
    
    X_future = future_with_features[feature_cols]
    
    # 8. Verificar datos nulos
    nan_count = X_future.isna().sum().sum()
    if nan_count > 0:
        numeric_cols = X_future.select_dtypes(include=[np.number]).columns
        X_future[numeric_cols] = X_future[numeric_cols].fillna(0)
        
    # 9. Predicción
    predictions = model.predict(X_future)
    predictions = np.round(np.clip(predictions, 0, None)).astype(int)
    
    # 10. Generar resultado
    results = future_with_features[['date'] + group_cols].copy()
    results['predicted_sales'] = predictions
    
    print(f'\n✅ Predictions:')
    print(f'Total: {len(results):,}')
    print(f'Range: [{predictions.min():.2f}, {predictions.max():.2f}]')
    print(f'Mean: {predictions.mean():.2f}')
    print(f'Total estimated sales: {predictions.sum():.2f}')
    
    return results



def dataset_for_dashboard(train, test, future_predictions):
    if test is not None and 'sales' in test.columns:
        historical = pd.concat([train, test], ignore_index=True)
    else:
        historical = train.copy()
        
    hist = historical[['date', 'store', 'item', 'sales']].copy()
    hist['type'] = 'historical'
    hist['predicted_sales'] = None
    hist.rename(columns={'sales': 'actual_sales'}, inplace=True)
    
    pred = future_predictions.copy()
    pred['type'] = 'forecast'
    pred['actual_sales'] = None
    
    pred = pred[['date', 'store', 'item', 'type', 'actual_sales', 'predicted_sales']]
    hist = hist[['date', 'store', 'item', 'type', 'actual_sales', 'predicted_sales']]
    
    looker_data = pd.concat([hist, pred], ignore_index=True)
    looker_data = looker_data.sort_values(['store', 'item', 'date']).reset_index(drop=True)
    
    print(f'✅ Dataset for dashboard:')
    print(f'Total rows: {len(looker_data):,}')
    print(f'Historic: {(looker_data['type']=='historical').sum():,}')
    print(f'Forecast: {(looker_data['type']=='forecast').sum():,}')
    
    return looker_data
    
    