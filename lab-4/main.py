def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, capacity + 1):
            if weights[i - 1] <= j:
                dp[i][j] = max(dp[i - 1][j], values[i - 1] + dp[i - 1][j - weights[i - 1]])
            else:
                dp[i][j] = dp[i - 1][j]

    selected_items = []
    i, j = n, capacity
    while i > 0 and j > 0:
        if dp[i][j] != dp[i - 1][j]:
            selected_items.append(i - 1)
            j -= weights[i - 1]
        i -= 1

    selected_items.reverse()
    return dp[n][capacity], selected_items


# Пример использования
weights = [2, 3, 4, 5]  # веса предметов
values = [3, 4, 5, 6]  # значения (пользы) предметов
capacity = 8  # вместимость рюкзака

max_value, selected_items = knapsack(weights, values, capacity)

print("Максимальная польза:", max_value)
print("Выбранные предметы (индексы):", selected_items)