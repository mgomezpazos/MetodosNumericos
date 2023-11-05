import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Carga el conjunto de datos
data = pd.read_csv('dataset02.csv')
X = data.values  # Convierte los datos a una matriz NumPy

# Valores de d a considerar
dims = [2, 4, 6, 20, X.shape[1]]  # Incluye la dimensionalidad original (p)

# Definir el valor de sigma para calcular la similitud
sigma = 1.0  # Puedes ajustar este valor

# Realiza la reducción de dimensionalidad y análisis
for d in dims:
    # Calcula la SVD de X
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    
    # Reduce la dimensionalidad seleccionando las d componentes principales
    X_reduced = np.dot(U[:, :d], np.dot(np.diag(S[:d]), VT[:d, :]))
    
    # Plotea las proyecciones en el nuevo espacio reducido
    plt.figure(figsize=(6, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c='b', marker='o', alpha=0.5)
    plt.title(f'Dimensión reducida d={d}')
    plt.show()

# Similitud entre muestras en el espacio original y reducido
for d in dims:
    # Calcula la similitud en el espacio original
    similarity_original = np.exp(-np.sum((X[:, None, :] - X) ** 2, axis=2) / (2.0 * sigma**2))
    
    # Calcula la similitud en el espacio reducido (usando SVD)
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    X_reduced = np.dot(U[:, :d], np.dot(np.diag(S[:d]), VT[:d, :]))
    similarity_reduced = np.exp(-np.sum((X_reduced[:, None, :] - X_reduced) ** 2, axis=2) / (2.0 * sigma**2))
    
    # Plotea las similitudes como gráficos de dispersión
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c='b', marker='o', alpha=0.5)
    plt.title('Similitud Espacio Original')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c='b', marker='o', alpha=0.5)
    plt.title(f'Similitud Espacio Reducido (d={d})')
    
    plt.show()


#2.2---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Muestra las dimensiones más representativas con respecto a las dimensiones reducidas obtenidas por SVD
for d in dims:
    # Selecciona las dimensiones VT correspondientes a la reducción d
    VT_reduced = VT[:d, :]

    # Calcula el promedio de los valores absolutos de cada dimensión de VT_reduced
    avg_abs = np.mean(np.abs(VT_reduced), axis=1)

    # Ordena las dimensiones en orden descendente según su importancia
    sorted_dims = np.argsort(avg_abs)[::-1]

    print(f"Dimensiones más importantes para d={d} según SVD:")
    for i in range(min(5, len(sorted_dims))):  # Muestra las 5 dimensiones más importantes o las disponibles
        print(f"Dimensión {sorted_dims[i]} - Importancia: {avg_abs[sorted_dims[i]]}")

    print()


#2.3-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Carga de datos del dataset X y las etiquetas Y
X = pd.read_csv('dataset02.csv').values
Y = pd.read_csv('y2.txt').values
Y = Y.reshape(-1, 1)  # Asegura que Y sea de dimensiones (n, 1)

# Encuentra la fila problemática (si las dimensiones no coinciden)
if X.shape[0] != Y.shape[0]:
    # Identifica la fila que está causando la incompatibilidad
    problem_row_index = None
    if X.shape[0] > Y.shape[0]:
        problem_row_index = X.shape[0] - 1  # Última fila de X
    else:
        problem_row_index = Y.shape[0] - 1  # Última fila de Y

    # Elimina la fila problemática de la matriz X
    X = np.delete(X, problem_row_index, axis=0)

# Continuar con el análisis con las dimensiones compatibles entre X y Y
dims = [2, 5, 10, 20]  # Define las dimensiones a probar

min_error = float('inf')
best_d = 0

for d in dims:
    # Calcula la SVD de X
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    
    # Reduce la dimensionalidad seleccionando las d componentes principales
    X_reduced = np.dot(U[:, :d], np.dot(np.diag(S[:d]), VT[:d, :]))

    beta = np.linalg.lstsq(X_reduced, Y, rcond=None)[0]
    
    # Calcula las predicciones
    predictions = np.dot(X_reduced, beta)
    
    # Calcula el error de predicción
    error = np.linalg.norm(predictions - Y)**2
    
    # Guarda el mejor d basado en el error
    if error < min_error:
        min_error = error
        best_d = d

    # Plotea las proyecciones en el nuevo espacio reducido
    plt.figure(figsize=(6, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c='b', marker='o', alpha=0.5)
    plt.title(f'Dimensión reducida d={d}')
    plt.show()

# Muestra el mejor d y el error mínimo
print(f"La mejor d que minimiza el error es: {best_d}")
print(f"El mínimo error es: {min_error}")