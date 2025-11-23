import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def plot_dashboard(y_true, y_pred, n_samples=300, n_bins_residuals=50, 
                    n_bins_distribution=30, n_ranges=5, figsize=(16, 10)):
    """
        Crea un dashboard con los resultados de validación de un determinado modelo.
        Incluye:
            • Comparación entre series de tiempo.
            • Gráfico de dispersión.
            • Análisis residual.
            • Sumario de métricas.
            • Error por rango de valores.
            • Comparación de distribuciones.
        
        Parámetros
        ----------
        y_true : array
            Valores reales del dataset de validación.
        y_pred : array
            Predicciones del modelo.
        n_samples : int, default=300
            Número de muestras para mostrar en las series de tiempo.
        n_bins_residuals : int, default=50
            Número de cajas para el histograma de resíduos.
        n_bins_distribution : int, default=30
            Número de cajas para el histograma de comparación.
        n_ranges : int, default=5
            Valor de rangos para el análisis de error.
        figsize : tuple, default=(16, 10)
            Tamaño del dashboard (largo, alto)
    """
    
    # Convertir a arrays de NumPy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calcular métricas
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # Creación de figura con grillas
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)
    
    # =====================================================
    # 1. Series Temporales
    # =====================================================
    ax1 = fig.add_subplot(gs[0, :])
    samples_to_plot = min(n_samples, len(y_true))
    ax1.plot(y_true[:samples_to_plot], label='Real', alpha=0.8, linewidth=1.5)
    ax1.plot(y_pred[:samples_to_plot], label='Predicción', alpha=0.8, linewidth=1.5)
    ax1.set_title(f'Real vs Predicción (Primeras {samples_to_plot} muestras)', 
                  fontweight='bold', loc='left')
    ax1.set_xlabel('Muestras', fontweight='bold')
    ax1.set_ylabel('Valores', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # =====================================================
    # 2. Gráfico de Dispersión
    # =====================================================
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(y_true, y_pred, alpha=0.2, s=5)
    
    # Línea de predicción perfecta
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
    
    ax2.set_xlabel('Real', fontweight='bold')
    ax2.set_ylabel('Predicción', fontweight='bold')
    ax2.set_title('Scatter Plot', fontweight='bold', loc='left')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # =====================================================
    # 3. Distribución de Resíduos
    # =====================================================
    ax3 = fig.add_subplot(gs[1, 1])
    residuos = y_true - y_pred
    ax3.hist(residuos, bins=n_bins_residuals, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.axvline(x=0, color='r', linestyle='--', lw=2, label='Cero')
    ax3.set_xlabel('Residuos', fontweight='bold')
    ax3.set_ylabel('Frecuencia', fontweight='bold')
    ax3.set_title('Distribución de Residuos', fontweight='bold', loc='left')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # =====================================================
    # 4. Sumario de Métricas
    # =====================================================
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    
    metrics_text = f"""MÉTRICAS DE VALIDACIÓN
    MAE:  {mae:.2f}
    RMSE: {rmse:.2f}
    R²:   {r2:.4f}
    MAPE: {mape:.2f}%

    Total muestras: {len(y_true):,}"""
    
    ax4.text(0.0, 0.5, metrics_text, fontsize=14, 
             verticalalignment='center', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', 
                      alpha=0.7, edgecolor='black', linewidth=1.5),
             family='monospace', fontweight='bold',
             transform=ax4.transAxes)
    
    # =====================================================
    # 5. Error por Rango de Valores
    # =====================================================
    ax5 = fig.add_subplot(gs[2, :2])
    
    val_df_temp = pd.DataFrame({'real': y_true, 'pred': y_pred})
    val_df_temp['rango'] = pd.cut(val_df_temp['real'], bins=n_ranges)
    
    error_por_rango = val_df_temp.groupby('rango', observed=True).apply(
        lambda x: np.mean(np.abs(x['real'] - x['pred'])),
        include_groups=False
    )
    
    error_por_rango.plot(kind='bar', ax=ax5, color='coral', edgecolor='black')
    ax5.set_xlabel('Rango de Valores Reales', fontweight='bold')
    ax5.set_ylabel('MAE', fontweight='bold')
    ax5.set_title('Error por Rango de Valores', fontweight='bold', loc='left')
    ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # =====================================================
    # 6. Comparación de Distribuciones
    # =====================================================
    ax6 = fig.add_subplot(gs[2, 2])
    
    ax6.hist(y_true, bins=n_bins_distribution, alpha=0.5, label='Real', 
             edgecolor='black', color='blue')
    ax6.hist(y_pred, bins=n_bins_distribution, alpha=0.5, label='Predicción', 
             edgecolor='black', color='orange')
    
    ax6.set_xlabel('Valores', fontweight='bold')
    ax6.set_ylabel('Frecuencia', fontweight='bold')
    ax6.set_title('Distribución de Valores', fontweight='bold', loc='left')
    ax6.legend()
    ax6.grid(True, alpha=0.3, linestyle='--')
    
    # =====================================================
    # Título General
    # =====================================================
    # plt.suptitle('Dashboard de Validación del Modelo', fontsize=16, fontweight='bold', y=0.98)
    
    plt.show()
    
    