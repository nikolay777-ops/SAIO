import numpy as np

# Матрица прибыли
A = np.array([
    [0, 1, 2, 3],
    [0, 0, 1, 2],
    [0, 2, 2, 3]
])

P = 3
Q = 3  # Количество агентов и ресурсов

# Создаем массив для хранения результатов
dp = np.zeros((P, Q + 1))

# Заполняем dp массив снизу вверх
for i in range(P - 1, -1, -1):
    for j in range(Q + 1):
        max_profit = 0
        for k in range(j + 1):
            current_profit = A[i, k]
            if i < P - 1:
                current_profit += dp[i + 1, j - k]
            max_profit = max(max_profit, current_profit)
        dp[i, j] = max_profit

# Максимальная прибыль равна значению в верхнем левом углу dp массива
max_profit = dp[0, Q]
print("Максимальная прибыль:", max_profit)
