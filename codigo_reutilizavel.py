import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone

# ==================== 1. Carregamento e Preparo ==================== #

def carregar_dados(caminho_area, caminho_caracteristicas):

    # Carregar dados
    dfA = pd.read_excel(caminho_area)
    dfC = pd.read_excel(caminho_caracteristicas)

    # Limpar colunas irrelevantes
    dfC_clean = dfC.drop(columns=["attributes", "samples", "dimension"], errors="ignore")

    # Unir DataFrames
    df = pd.concat([dfA, dfC_clean], axis=1)

    # Variável resposta e preditoras
    y = df["Area(mm²)"]
    X = df.drop(columns=["Area(mm²)", "identificador"], errors="ignore")

    # Criar variável de binning para estratificação (10 quantis)
    y_binned = pd.qcut(y, q=10, labels=False)

    return X, y, y_binned

# ==================== 2. Treinamento e Avaliação ==================== #

def treinar_e_avaliar_modelo(X, y, y_binned, modelo, plot=True):

    # === Pipeline de modelagem ===
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", modelo)
    ])

    # Estratégia de validação cruzada estratificada
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    # Inicializar vetor de predições
    y_pred = np.zeros_like(y, dtype=float)

    # === Validação cruzada manual com estratificação ===
    for train_idx, test_idx in cv.split(X, y_binned):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        model = clone(pipeline)
        model.fit(X_train, y_train)
        y_pred[test_idx] = model.predict(X_test)

    # Avaliação
    rmse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"Modelo: {modelo.__class__.__name__}")
    print(f"R²: {r2:.3f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    # === Gráfico: Observado vs Predito ===
    if plot:
        plt.figure(figsize=(7, 6))
        sns.scatterplot(x=y, y=y_pred, alpha=0.6)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.xlabel("Área Observada (mm²)")
        plt.ylabel("Área Predita (mm²)")
        plt.title(f"Observado vs Predito - {modelo.__class__.__name__}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return r2, rmse, mae

# ==================== 3. Exemplo de Uso ==================== #

# from sklearn.linear_model import Ridge
# # Você pode importar e testar qualquer outro modelo também:
# # from sklearn.ensemble import RandomForestRegressor

# X, y, y_binned = carregar_dados("area.xlsx", "caracteristicas.xlsx")
# modelo = Ridge(alpha=30.5)

# treinar_e_avaliar_modelo(X, y, y_binned, modelo)

# ================= Testando todos os modelos ==================

from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin
import warnings

warnings.filterwarnings("ignore")  # silenciar avisos de convergência, etc.

X, y, y_binned = carregar_dados("area.xlsx", "caracteristicas.xlsx")

resultados = []

for nome, ClasseModelo in all_estimators(type_filter='regressor'):
    try:
        modelo = ClasseModelo()  # tenta instanciar com os defaults
        print(f"\nTestando modelo: {nome}")
        r2, rmse, mae = treinar_e_avaliar_modelo(X, y, y_binned, modelo, plot=False)
        resultados.append({
            "Modelo": nome,
            "R²": r2,
            "RMSE": rmse,
            "MAE": mae
        })
    except Exception as e:
        print(f"⚠️ Erro ao testar {nome}: {e}")