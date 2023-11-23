import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt

n = 5
d = 100

A = np.random.rand(n, d)
b = np.random.rand(n)

# Función de costo F(x)
def cost_function_F(x):
    return np.linalg.norm(np.dot(A, x) - b)**2

# Función de costo F2(x) con regularización L2
def cost_function_F2(x, delta_squared):
    return cost_function_F(x) + delta_squared * np.linalg.norm(x)**2

def gradient_descent(cost_function, initial_guess, learning_rate, iterations, delta_squared=None):
    x = initial_guess.copy()
    for i in range(iterations):
        gradient = 2 * np.dot(A.T, (np.dot(A, x) - b))
        if delta_squared is not None:
            gradient += 2 * delta_squared * x
        gradient /= np.linalg.norm(gradient)
        x -= learning_rate * gradient
    return x

# Obtención de la solución mediante SVD
U, S, Vt = svd(A, full_matrices=False)
V = Vt.T
pseudoinv = np.dot(V, np.dot(np.diag(1/S), U.T))
solution_svd = np.dot(pseudoinv, b)

# Parámetros
learning_rate = 1 / S[0]  # Usando el inverso del mayor valor singular para el tamaño de paso
iterations = 1000
delta_squared = 1e-2 * np.max(S)  # Valor delta^2 para F2

# Condición inicial aleatoria
initial_guess = np.random.rand(d)

solution_F = gradient_descent(cost_function_F, initial_guess, learning_rate, iterations)
solution_F2 = gradient_descent(cost_function_F2, initial_guess, learning_rate, iterations, delta_squared)


print("Solución mediante SVD:", solution_svd)
print("Solución mediante gradiente descendiente (F):", solution_F)
print("Solución mediante gradiente descendiente (F2):", solution_F2)


def gradient_descent_tracking(cost_function, initial_guess, learning_rate, iterations, delta_squared=None):
    x = initial_guess.copy()
    solutions = [x.copy()]  # Almacenar la solución inicial
    for i in range(iterations):
        gradient = 2 * np.dot(A.T, (np.dot(A, x) - b))
        if delta_squared is not None:
            gradient += 2 * delta_squared * x
        gradient /= np.linalg.norm(gradient)  # Normalizar el gradiente
        x -= learning_rate * gradient
        solutions.append(x.copy())  # Almacenar la solución actual
    return solutions

# Obtener soluciones a lo largo de las iteraciones
solutions_F = gradient_descent_tracking(cost_function_F, initial_guess, learning_rate, iterations)
solutions_F2 = gradient_descent_tracking(cost_function_F2, initial_guess, learning_rate, iterations, delta_squared)


num_iterations = range(iterations + 1)  # Iteraciones + solución inicial

# Obtener los valores de la función de costo a lo largo de las iteraciones para F y F2
cost_values_F = [cost_function_F(sol) for sol in solutions_F]
cost_values_F2 = [cost_function_F2(sol, delta_squared) for sol in solutions_F2]

# Obtener los valores de la función de costo a lo largo de las iteraciones para SVD con F y F2
cost_values_svd = [cost_function_F(solution_svd)] * (iterations + 1)
cost_values_svd2 = [cost_function_F2(solution_svd, delta_squared)] * (iterations + 1)

# Graficar la función de costo a lo largo de las iteraciones para F, F2 y SVD
plt.figure(figsize=(10, 6))
plt.title("Valor de la función de costo a lo largo de las iteraciones")
plt.xlabel("Iteración")
plt.ylabel("Valor de la función de costo")
plt.plot(num_iterations, cost_values_F, label="Costo (F)", alpha=0.7)
plt.plot(num_iterations, cost_values_F2, label="Costo (F2)", alpha=0.7)
plt.plot(num_iterations, cost_values_svd, label="Costo (SVD)", alpha=0.7)
plt.plot(num_iterations, cost_values_svd2, label="Costo (SVD2)", alpha=0.7)
plt.legend()
plt.show()

#Error ---------------------------------------------------------------------------------------------------------------------------------------------------
# Calcular el error respecto al mínimo encontrado por el método
error_F = np.array(cost_values_F) - cost_function_F(solution_F)
error_F2 = np.array(cost_values_F2) - cost_function_F2(solution_F2, delta_squared)
error_svd = np.array(cost_values_svd) - cost_function_F(solution_svd)
error_svd2 = np.array(cost_values_svd2) - cost_function_F2(solution_svd, delta_squared)

# Graficar la convergencia del error para F, F2 y SVD
plt.figure(figsize=(10, 6))
plt.title("Convergencia del error (F, F2, SVD)")
plt.xlabel("Iteración")
plt.ylabel("Error")
plt.plot(num_iterations, error_F, label="Error (F)", alpha=0.7)
plt.plot(num_iterations, error_F2, label="Error (F2)", alpha=0.7)
error_svd = np.array(cost_values_svd) - cost_function_F(solution_svd)
error_svd2 = np.array(cost_values_svd2) - cost_function_F2(solution_svd, delta_squared)
plt.plot(num_iterations, error_svd, label="Error (SVD)", alpha=0.7)
plt.legend()
plt.show()


# Variando delta------------------------------------------------------------------------------------------------------------------------------------------
delta_squared_values = [1e-4, 1e-2, 1,50,100,1000,15000]  # Valores de delta_squared a probar

errors_F2_varied = []  # Almacenar los errores para cada valor de delta_squared
solutions_F2_varied = []  # Almacenar las soluciones para cada valor de delta_squared

for delta_squared_val in delta_squared_values:
    solution_F2_varied = gradient_descent(cost_function_F2, initial_guess, learning_rate, iterations, delta_squared_val)
    solutions_F2_varied.append(solution_F2_varied)

    # Calcular los errores respecto al mínimo encontrado por el método para F2 variado
    cost_values_F2_varied = [cost_function_F2(sol, delta_squared_val) for sol in gradient_descent_tracking(
        cost_function_F2, initial_guess, learning_rate, iterations, delta_squared_val)]
    errors_F2_varied.append(np.array(cost_values_F2_varied) - cost_function_F2(solution_F2_varied, delta_squared_val))

# Graficar la convergencia del error para F2 con diferentes delta_squared
plt.figure(figsize=(10, 6))
plt.title("Convergencia del error con diferentes delta_squared (F2)")
plt.xlabel("Iteración")
plt.ylabel("Error")

for i, delta_squared_val in enumerate(delta_squared_values):
    plt.plot(range(iterations + 1), errors_F2_varied[i], label=f"delta_squared = {delta_squared_val}", alpha=0.7)

plt.legend()
plt.show()


#Variando S -------------------------------------------------------------------------------------------------------------
# Definir los valores para S en un rango alrededor de 1/λmax
S_values = np.linspace(1/(2 * S[0]), 2 * (1/S[0]), num=5)  # 10 valores entre 1/(2*λmax) y 2*(1/λmax)

errors_S_varied = []  # Almacenar los errores para cada valor de S
solutions_S_varied = []  # Almacenar las soluciones para cada valor de S

for S_val in S_values:
    # Calcular el learning rate basado en el valor de S
    learning_rate_S = S_val

    # Ejecutar el gradiente descendente con el nuevo learning rate
    solution_S_val = gradient_descent(cost_function_F, initial_guess, learning_rate_S, iterations)

    # Almacenar la solución y calcular el error
    solutions_S_varied.append(solution_S_val)
    cost_values_S_val = [cost_function_F(sol) for sol in gradient_descent_tracking(
        cost_function_F, initial_guess, learning_rate_S, iterations)]
    errors_S_varied.append(np.array(cost_values_S_val) - cost_function_F(solution_S_val))

# Graficar la convergencia del error para diferentes valores de S
plt.figure(figsize=(10, 6))
plt.title("Convergencia del error con diferentes valores de S")
plt.xlabel("Iteración")
plt.ylabel("Error")

for i, S_val in enumerate(S_values):
    plt.plot(range(iterations + 1), errors_S_varied[i], label=f"S = {S_val:.3f}", alpha=0.7)

plt.legend()
plt.show()