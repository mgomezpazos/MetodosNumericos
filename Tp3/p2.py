import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Carga el conjunto de datos
data = pd.read_csv('dataset02.csv')
X = data.values  # Convierte los datos a una matriz NumPy

# Valores de d a considerar
dims = [2, 4, 6, 20, X.shape[1]]  # Incluye la dimensionalidad original (p)

# Definir el valor de sigma para calcular la similitud
sigma = 1.0  # Puedes ajustar este valor

# Gráfico para todas las proyecciones
plt.figure(figsize=(15, 10))

for i, d in enumerate(dims, 1):
    # Calcula la SVD de X
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    
    # Reduce la dimensionalidad seleccionando las d componentes principales
    X_reduced = np.dot(U[:, :d], np.dot(np.diag(S[:d]), VT[:d, :]))
    
    plt.subplot(2, 3, i)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c='violet', marker='o', alpha=0.5)
    plt.xlabel('Componente principal 1')
    plt.ylabel('Componente principal 2')

plt.tight_layout(w_pad=4, h_pad=4)
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Calcula la similitud en el espacio original
similarity_original = np.exp(-np.sum((X[:, None, :] - X) ** 2, axis=2) / (2.0 * sigma**2))
axs[0, 0].scatter(X[:, 0], X[:, 1], c='violet', marker='o', alpha=0.5)
axs[0, 0].set_xlabel('Dimensión 1')
axs[0, 0].set_ylabel('Dimensión 2')

# Muestra las comparaciones para diferentes d utilizando SVD
for i, d in enumerate([2, 4, 6, 20, 107], 1):
    # Calcula la similitud en el espacio reducido (usando SVD)
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    X_reduced = np.dot(U[:, :d], np.dot(np.diag(S[:d]), VT[:d, :]))
    similarity_reduced = np.exp(-np.sum((X_reduced[:, None, :] - X_reduced) ** 2, axis=2) / (2.0 * sigma**2))

    axs[i // 3, i % 3].scatter(X_reduced[:, 0], X_reduced[:, 1], c='violet', marker='o', alpha=0.5)
    axs[i // 3, i % 3].set_xlabel('Componente principal 1')
    axs[i // 3, i % 3].set_ylabel('Componente principal 2')

plt.tight_layout()
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

for i, d in enumerate(dims, 1):
    # Calcula la SVD de X
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    
    # Reduce la dimensionalidad seleccionando las d componentes principales
    X_reduced = np.dot(U[:, :d], np.dot(np.diag(S[:d]), VT[:d, :]))

    beta = np.linalg.lstsq(X_reduced, Y, rcond=None)[0]
    
    # Calcula las predicciones
    predictions = np.dot(X_reduced, beta)
    
    # Calcula el error de predicción
    error = np.linalg.norm(predictions - Y) ** 2
    
    # Guarda el mejor d basado en el error
    if error < min_error:
        min_error = error
        best_d = d

# Muestra el mejor d y el error mínimo
print(f"La mejor d que minimiza el error es: {best_d}")
print(f"El mínimo error es: {min_error}")


errors = []  # Almacenar los errores para cada d
for d in dims:
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    X_reduced = np.dot(U[:, :d], np.dot(np.diag(S[:d]), VT[:d, :]))
    beta = np.linalg.lstsq(X_reduced, Y, rcond=None)[0]
    predictions = np.dot(X_reduced, beta)
    error = np.linalg.norm(predictions - Y)**2
    errors.append(error)

# Gráfico de error vs. Dimensionalidad
plt.figure(figsize=(8, 6))
plt.plot(dims, errors, marker='o')
plt.xlabel('Dimensionalidad (d)')
plt.ylabel('Error de Predicción')
plt.show()

errors = []  # Almacenar los errores para cada d
for d in dims:
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    X_reduced = np.dot(U[:, :d], np.dot(np.diag(S[:d]), VT[:d, :]))
    beta = np.linalg.lstsq(X_reduced, Y, rcond=None)[0]
    predictions = np.dot(X_reduced, beta)
    error = np.linalg.norm(predictions - Y) ** 2
    errors.append(error)

# Gráfico de histogramas para los errores
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=10, alpha=0.7, color='skyblue')
plt.xlabel('Error de Predicción')
plt.ylabel('Frecuencia')
plt.show()

betas = []  # Almacenar los coeficientes beta para cada d
for d in dims:
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    X_reduced = np.dot(U[:, :d], np.dot(np.diag(S[:d]), VT[:d, :]))
    beta = np.linalg.lstsq(X_reduced, Y, rcond=None)[0]
    betas.append(beta)

# Crear un gráfico de barras para mostrar los coeficientes beta
num_attributes = X.shape[1]
plt.figure(figsize=(12, 6))

for i, beta in enumerate(betas):
    plt.bar(np.arange(num_attributes) + i * 0.2, beta.flatten(), width=0.2, label=f'd={dims[i]}')
plt.xlabel('Atributos')
plt.ylabel('Valor del Coeficiente beta')
plt.legend()
plt.show()

# Suponiendo que tienes X_reduced para diferentes valores de d
dims = [2, 4, 6]  # Valores de d
data = {}  # Un diccionario para almacenar los datos de X_reduced

# Genera datos de ejemplo (reemplaza esto con tus datos reales)
for d in dims:
    # Genera datos aleatorios para X_reduced (solo como ejemplo)
    n_samples = 1000
    X_reduced = np.random.rand(n_samples, d)  # Suponiendo que 'X_reduced' es una matriz de forma (n_samples, d)
    data[f'd={d}'] = X_reduced

# Ploteo de gráficos de densidad
fig, axs = plt.subplots(1, len(dims), figsize=(15, 5))

for i, d in enumerate(dims):
    # Gráfico de densidad
    sns.kdeplot(data=data[f'd={d}'][:, 0], fill=True, color='blue', ax=axs[i])
    sns.kdeplot(data=data[f'd={d}'][:, 1], fill=True, color='red', ax=axs[i])
    axs[i].set_xlabel('Valores')
    axs[i].set_ylabel('Densidad')

plt.tight_layout()
plt.show()