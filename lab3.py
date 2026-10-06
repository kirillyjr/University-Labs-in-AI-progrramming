import random

# Импортируем весь набор функций из своего модуля feature_processing.py
# (задание 4 - вынести логику в отдельный модуль и переиспользовать её)
from feature_processing import *




def fmt_list(values, n=10, digits=3):
    """Округляет и обрезает список чисел для компактного вывода."""
    return [round(v, digits) for v in values[:n]]


def line(label, value):
    """Печатает строку 'подпись: значение' с выравниванием по колонке."""
    print(f"{label:<{28}}{value}")


def analyze_dataset(data, name):
    """
    Прогоняет один набор данных через все функции модуля.
    Вынесено в отдельную функцию, чтобы не дублировать один и тот же
    код анализа для каждого набора данных (демонстрация DRY / reuse).
    """
    print(f"\n{'=' * 70}")
    print(name)
    print(f"{'=' * 70}")
    line("Данные (первые 10):", fmt_list(data))

    # Базовые статистики - задание 1
    print("\n--- Статистики ---")
    line("Среднее:", f"{mean(data):.3f}")
    line("Медиана:", f"{median(data):.3f}")
    line("Дисперсия:", f"{variance(data):.3f}")
    line("Стандартное отклонение:", f"{std(data):.3f}")

    # Предобработка данных - задание 2
    print("\n--- Предобработка ---")
    scaled = min_max_scale(data, (0, 1))
    line("Min-max scale (первые 10):", fmt_list(scaled))

    logged = log_scale(data)
    line("Log scale (первые 10):", fmt_list(logged))

    standardized = standardize(data)
    line("Standardize (первые 10):", fmt_list(standardized))

    cleaned = remove_outliers(data, k=2)
    line("Remove outliers (k=2):", f"было {len(data)}, осталось {len(cleaned)}")

    # Конвейер преобразований - задание 3
    # Каждая функция получает результат предыдущей: сначала стандартизация,
    # потом приведение уже стандартизованных значений к диапазону [0, 1]
    print("\n--- Конвейер: standardize -> min_max_scale ---")
    pipeline = [standardize, lambda x: min_max_scale(x, (0, 1))]
    processed = apply_pipeline(data, pipeline)
    line("Результат (первые 10):", fmt_list(processed))


# Два разных набора данных для демонстрации повторного использования
# одной и той же функции analyze_dataset (задание 4)
data_uniform = [random.randint(1, 10000) for _ in range(100)]
data_gauss = [round(random.gauss(500, 50)) for _ in range(120)]

analyze_dataset(data_uniform, "Набор 1: равномерное распределение [1, 10000]")
analyze_dataset(data_gauss, "Набор 2: нормальное распределение N(500, 50)")
