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


#1.3-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Carga de datos etiquetas
Y = pd.read_csv('y2.txt').values
Y = Y.T  # Transpone Y para ajustar las dimensiones

# Realiza la reducción de dimensionalidad y análisis
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