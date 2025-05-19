import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# =============================================================================
# 1. Função para remover features com alta correlação entre si
# =============================================================================
def remover_features_correlacionadas(df, limite, verbose=True, exibir_matriz=False):
    df_numerico = df.select_dtypes(include=["number"])
    matriz_corr = df_numerico.corr()
    colunas_correlacionadas = set()

    for i in range(len(matriz_corr.columns)):
        for j in range(i):
            if abs(matriz_corr.iloc[i, j]) > limite:
                colunas_correlacionadas.add(matriz_corr.columns[i])

    colunas_mantidas = [col for col in df.columns if col not in colunas_correlacionadas]

    if verbose:
        if colunas_correlacionadas:
            print(f"[ℹ] Colunas removidas por correlação > {limite}: {sorted(colunas_correlacionadas)}")
        else:
            print(f"[✔] Nenhuma coluna removida por correlação > {limite}")
        
        print(f"[✅] {len(colunas_mantidas)} colunas restantes para uso no treinamento:")
        print(sorted(colunas_mantidas))

    if exibir_matriz:
        plt.figure(figsize=(10, 8))
        sns.heatmap(matriz_corr, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
        plt.title("Matriz de Correlação")
        plt.tight_layout()
        plt.show()

    return df.drop(columns=colunas_correlacionadas, errors='ignore')

# =============================================================================
# 2. Preparação de Dados
# =============================================================================
def preparar_dados(limite_correlacao, verbose=True):
    """
    Prepara os dados aplicando apenas a remoção de colunas com alta correlação entre si.

    Parâmetros:
        limiar_correlacao (float): Limite acima do qual colunas altamente correlacionadas são removidas.
        verbose (bool): Exibe colunas removidas e mantidas se True.

    Retorna:
        X (pd.DataFrame): Conjunto de features numéricas sem colunas altamente correlacionadas.
        y (pd.Series): Variável alvo (Área).
        y_binned (pd.Series): Target discretizado em bins para stratified KFold.
    """
    # 1. Carrega os dados
    dfA = pd.read_excel("./area.xlsx")
    dfC = pd.read_excel("./caracteristicas.xlsx")

    # Remover colunas indesejadas do DataFrame dfC
    dfC_clean = dfC.drop(columns=['attributes', 'samples', 'dimension'])

    # Unir DataFrames
    df = pd.concat([dfA, dfC_clean], axis=1)

    # Variável resposta e preditoras
    y = df["Area(mm²)"]

    X = df.drop(columns=["Area(mm²)", "identificador"], errors="ignore")
    X_correlationFix = remover_features_correlacionadas(X, limite_correlacao, verbose)

    # Criar variável de binning para estratificação (10 quantis)
    y_binned = pd.qcut(y, q=10, labels=False)

    return X_correlationFix, y, y_binned

# =============================================================================
# 3. Treinamento com validação cruzada
# =============================================================================
def treinar_com_crossval(X, y, y_binned, modelo):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", modelo)
    ])

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    y_pred = np.zeros_like(y, dtype=float)

    for train_idx, test_idx in cv.split(X, y_binned):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        model_cv = clone(pipeline)
        model_cv.fit(X_train, y_train)
        y_pred[test_idx] = model_cv.predict(X_test)

    return y_pred

# =============================================================================
# 4. Avaliação do modelo
# =============================================================================
def avaliar_modelo(y_real, y_predito, nome_modelo="", plot=True):
    r2 = r2_score(y_real, y_predito)
    rmse = mean_squared_error(y_real, y_predito, squared=False)
    mae = mean_absolute_error(y_real, y_predito)

    print(f"\n[📊 Avaliação - {nome_modelo}]")
    print(f"R²   : {r2:.3f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")

    if plot:
        plt.figure(figsize=(7, 6))
        sns.scatterplot(x=y_real, y=y_predito, alpha=0.6)
        plt.plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--')
        plt.xlabel("Área Observada (mm²)")
        plt.ylabel("Área Predita (mm²)")
        plt.title(f"Observado vs Predito - {nome_modelo}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return r2, rmse, mae

# =============================================================================
# 5. Teste do modelo
# =============================================================================
def testar_modelo(modelo, limiar_correlacao):
    nome = modelo.__class__.__name__

    try:
        X, y, y_binned = preparar_dados(limiar_correlacao=limiar_correlacao)
        y_pred = treinar_com_crossval(X, y, y_binned, modelo)
        r2, rmse, mae = avaliar_modelo(y, y_pred, nome_modelo=nome, plot=True)

        return {
            "Modelo": nome,
            "Limiar Correlação": limiar_correlacao,
            "R²": r2,
            "RMSE": rmse,
            "MAE": mae
        }

    except Exception as e:
        print(f"[ERRO] {nome} | limiar {limiar_correlacao}: {e}")
        return None

# =============================================================================
# 6. Execução de exemplo
# =============================================================================
if __name__ == "__main__":
    from sklearn.linear_model import Ridge

    modelo = Ridge(alpha=30.5)
    resultado = testar_modelo(modelo, limiar_correlacao=0.8)

    if resultado:
        df_result = pd.DataFrame([resultado])
        df_result.to_excel("resultado_unico.xlsx", index=False)
        print("\n✅ Resultado salvo em 'resultado_unico.xlsx'")