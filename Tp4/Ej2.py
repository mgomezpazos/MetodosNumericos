import numpy as np
from scipy.optimize import minimize
import pandas as pd

# Leer datos desde el archivo CSV
data = pd.read_csv('retornos_historicos.csv', header=None)

# Aquí solo se proporciona una matriz de ejemplo para demostración, deberás cargar tus datos reales.
returns_matrix = np.random.rand(24, 100)

# Parámetros
r_min_values = [0.02, 0.03, 0.04]  # Diferentes valores de r_min

# Iterar sobre los diferentes valores de r_min
for r_min in r_min_values:
    # Calcular la matriz de covarianza
    cov_matrix = np.cov(returns_matrix, rowvar=False)

    # Vector de medias de los retornos
    r_mean = np.mean(returns_matrix, axis=0)

    # Función objetivo cuadrática
    def objective_function(x, cov_matrix):
        return x.T @ cov_matrix @ x

    # Restricciones
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Suma de las variables igual a 1
                   {'type': 'ineq', 'fun': lambda x: r_mean @ x - r_min})  # Restricción de riesgo mínimo

    # Condiciones iniciales
    x0 = np.ones(len(returns_matrix[0])) / len(returns_matrix[0])

    # Resolver el problema de optimización
    result = minimize(objective_function, x0, args=(cov_matrix,), constraints=constraints)

    # Mostrar los resultados para el valor actual de r_min
    print(f"\nMix óptimo de inversión para r_min = {r_min}:")
    print(result.x)