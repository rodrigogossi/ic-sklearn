# Instalar pacotes necessários (caso use Colab)
# !pip install openpyxl seaborn matplotlib scikit-learn pandas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone

# === 1. Carregamento e pré-processamento ===

# Carregar dados
dfA = pd.read_excel("area.xlsx")
dfC = pd.read_excel("caracteristicas.xlsx")

# Limpar colunas irrelevantes
dfC_clean = dfC.drop(columns=["attributes", "samples", "dimension"], errors="ignore")

# Unir DataFrames
df = pd.concat([dfA, dfC_clean], axis=1)

# Variável resposta e preditoras
y = df["Area(mm²)"]
X = df.drop(columns=["Area(mm²)", "identificador"], errors="ignore")

# Criar variável de binning para estratificação (10 quantis)
y_binned = pd.qcut(y, q=10, labels=False)

# === 2. Pipeline de modelagem ===

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=30.5))  # alpha otimizado previamente
])

# Estratégia de validação cruzada estratificada
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

# Inicializar vetor de predições
y_pred = np.zeros_like(y, dtype=float)

# === 3. Validação cruzada manual com estratificação ===

for train_idx, test_idx in cv.split(X, y_binned):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = y.iloc[train_idx]

    model = clone(pipeline)
    model.fit(X_train, y_train)
    y_pred[test_idx] = model.predict(X_test)

# === 4. Avaliação ===

rmse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"R²: {r2:.3f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# === 5. Gráfico: Observado vs Predito ===

plt.figure(figsize=(7, 6))
sns.scatterplot(x=y, y=y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Área Observada (mm²)")
plt.ylabel("Área Predita (mm²)")
plt.title("Observado vs Predito")
plt.grid(True)
plt.tight_layout()
plt.show()
