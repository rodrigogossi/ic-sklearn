from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin

# Lista todos os estimadores do tipo regressão
modelos = all_estimators(type_filter='regressor')

# Filtrar apenas os que realmente são RegressorMixin
modelos_regressao = [(nome, classe) for nome, classe in modelos if issubclass(classe, RegressorMixin)]

# Printar todos
print(f"Total de modelos de regressão encontrados: {len(modelos_regressao)}\n")
for nome, classe in modelos_regressao:
    print(nome)