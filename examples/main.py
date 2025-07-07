import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo limpio
sns.set(style='whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

def plot_comparison(df1, df2, label1, label2, metric_name, ylabel, filename):
    """
    Visualiza y guarda comparación entre dos modelos para una métrica dada,
    usando símbolo por tipo de métrica, color por modelo.
    """
    plt.figure(figsize=(10, 6))

    # Elegir símbolo según la métrica
    marker = 'o' if metric_name == 'loss' else 's'  # círculo para loss, cuadrado para accuracy

    # Colores por modelo
    color1 = '#1f77b4'  # MLP
    color2 = '#ff7f0e'  # CNN

    # Modelo 1
    plt.plot(df1['epoch'], df1[f'train_{metric_name}'], label=f'{label1} - Train', color=color1, marker=marker)
    plt.plot(df1['epoch'], df1[f'val_{metric_name}'], label=f'{label1} - Validation', color=color1, linestyle='--', marker=marker)

    # Modelo 2
    plt.plot(df2['epoch'], df2[f'train_{metric_name}'], label=f'{label2} - Train', color=color2, marker=marker)
    plt.plot(df2['epoch'], df2[f'val_{metric_name}'], label=f'{label2} - Validation', color=color2, linestyle='--', marker=marker)

    plt.title(f'{ylabel} Comparison: {label1} vs {label2}')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend(loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ Guardado: {filename}")

# Rutas a los CSV
csv_mlp = '../training_log_mlp.csv'
csv_cnn = '../training_log_cnn.csv'

# Cargar datos
df_mlp = pd.read_csv(csv_mlp)
df_cnn = pd.read_csv(csv_cnn)

# Comparar métricas con símbolo por tipo de métrica, color por modelo
plot_comparison(df_mlp, df_cnn, 'MLP', 'CNN', 'loss', 'Loss', 'loss_comparison.png')
plot_comparison(df_mlp, df_cnn, 'MLP', 'CNN', 'accuracy', 'Accuracy', 'accuracy_comparison.png')
