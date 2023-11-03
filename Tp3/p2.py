# import numpy as np
# import pandas as pd
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics.pairwise import rbf_kernel
# import matplotlib.pyplot as plt

# # Carga el conjunto de datos
# data = pd.read_csv('dataset02.csv')
# X = data.values  # Convierte los datos a una matriz NumPy

# # Valores de d a considerar
# dims = [2, 4, 6, 20, X.shape[1]]  # Incluye la dimensionalidad original (p)

# # Realiza la reducción de dimensionalidad para diferentes valores de d
# for d in dims:
#     # Aplica la descomposición SVD
#     svd = TruncatedSVD(n_components=d)
#     X_reduced = svd.fit_transform(X)
    
#     # Calcula las similitudes usando el kernel gaussiano
#     sigma = 1.0  # Puedes ajustar este valor
#     similarities = rbf_kernel(X_reduced, gamma=1.0 / (2.0 * sigma**2))

#     # Visualiza los datos en el nuevo espacio de dimensión reducida
#     plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c='b', marker='o', alpha=0.5)
#     plt.title(f'Dimensión reducida d={d}')
#     plt.show()

# # Similitud utilizando PCA y proyecciones al azar
# for d in dims:
#     # Utiliza PCA para reducir la dimensionalidad
#     pca = PCA(n_components=d)
#     X_pca = pca.fit_transform(X)
    
#     # Genera proyecciones al azar
#     num_random_projections = 100  # Puedes ajustar este valor
#     random_projections = np.random.randn(X.shape[1], d)
#     X_random = X.dot(random_projections)
    
#     # Calcula las similitudes utilizando el kernel gaussiano
#     sigma = 1.0  # Puedes ajustar este valor
#     similarities_pca = rbf_kernel(X_pca, gamma=1.0 / (2.0 * sigma**2))
#     similarities_random = rbf_kernel(X_random, gamma=1.0 / (2.0 * sigma**2))
    
#     # Plotea las similitudes
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.imshow(similarities_pca, cmap='hot', interpolation='none')
#     plt.title(f'Similitud PCA (d={d})')
#     plt.colorbar()
    
#     plt.subplot(1, 2, 2)
#     plt.imshow(similarities_random, cmap='hot', interpolation='none')
#     plt.title(f'Similitud Proyecciones al Azar (d={d})')
#     plt.colorbar()
    
#     plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Carga el conjunto de datos
data = pd.read_csv('dataset02.csv')
X = data.values  # Convierte los datos a una matriz NumPy

# Valores de d a considerar
dims = [2, 4, 6, 20, X.shape[1]]  # Incluye la dimensionalidad original (p)

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

#Esto para mi esta mal

# # Similitud utilizando PCA y proyecciones al azar
# for d in dims:
#     # Reducción de dimensionalidad con PCA sin scikit-learn
#     cov_matrix = np.cov(X, rowvar=False)
#     eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
#     sorted_indices = np.argsort(eigenvalues)[::-1]
#     eigenvalues = eigenvalues[sorted_indices]
#     selected_eigenvectors = eigenvectors[:, sorted_indices][:, :d]
#     P_pca = selected_eigenvectors
#     X_pca = X.dot(P_pca)
    
#     # Genera proyecciones al azar
#     num_random_projections = 100  # Puedes ajustar este valor
#     random_projections = np.random.randn(X.shape[1], d)
#     P_random = random_projections
#     X_random = X.dot(P_random)
    
#     # Calcula las similitudes utilizando el kernel gaussiano
#     sigma = 1.0  # Puedes ajustar este valor
#     similarities_pca = np.exp(-np.sum((X_pca[:, None, :] - X_pca) ** 2, axis=2) / (2.0 * sigma**2))
#     similarities_random = np.exp(-np.sum((X_random[:, None, :] - X_random) ** 2, axis=2) / (2.0 * sigma**2))
    
#     # Plotea las similitudes
#     plt.figure(figsize=(12, 6))
    
#     plt.subplot(1, 2, 1)
#     plt.imshow(similarities_pca, cmap='hot', interpolation='none')
#     plt.title(f'Similitud PCA (d={d})')
#     plt.colorbar()
    
#     plt.subplot(1, 2, 2)
#     plt.imshow(similarities_random, cmap='hot', interpolation='none')
#     plt.title(f'Similitud Proyecciones al Azar (d={d})')
#     plt.colorbar()
    
#     plt.show()

#Esto va mejor pero todavía hay que arreglarlo un toque

# Similitud utilizando PCA y proyecciones al azar
for d in dims:
    # Reducción de dimensionalidad con PCA sin scikit-learn
    cov_matrix = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    selected_eigenvectors = eigenvectors[:, sorted_indices][:, :d]
    P_pca = selected_eigenvectors
    X_pca = X.dot(P_pca)
    
    # Genera proyecciones al azar
    num_random_projections = 100  # Puedes ajustar este valor
    random_projections = np.random.randn(X.shape[1], d)
    P_random = random_projections
    X_random = X.dot(P_random)
    
    # Calcula las similitudes utilizando el kernel gaussiano
    sigma = 1.0  # Puedes ajustar este valor
    similarities_pca = np.exp(-np.sum((X_pca[:, None, :] - X_pca) ** 2, axis=2) / (2.0 * sigma**2))
    similarities_random = np.exp(-np.sum((X_random[:, None, :] - X_random) ** 2, axis=2) / (2.0 * sigma**2))
    
    # Plotea las similitudes como gráficos de dispersión
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c='b', marker='o', alpha=0.5)
    plt.title(f'Similitud PCA (d={d})')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_random[:, 0], X_random[:, 1], c='b', marker='o', alpha=0.5)
    plt.title(f'Similitud Proyecciones al Azar (d={d})')
    
    plt.show()