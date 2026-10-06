import math
import random


def validate(data: list) -> None:
    """
    Проверяет, что data - непустой список чисел.
    """
    if not isinstance(data, list):
        raise TypeError("data должен быть списком")
    if len(data) == 0:
        raise ValueError("data не должен быть пустым")
    if not all(isinstance(x, (int, float)) for x in data):
        # каждый элемент обязан быть числом, иначе дальше упадут sum/mean/сравнения
        raise TypeError("все элементы data должны быть числами")


def mean(data: list[float]) -> float:
    """
    Среднее арифметическое значение data.
    """
    validate(data)
    # среднее = сумма всех значений, делённая на их количество
    return sum(data) / len(data)


def quickselect(arr: list[float], k: int) -> float:
    """
    Возвращает k-й наименьший элемент (0-индексация) без полной сортировки.
    """
    if len(arr) == 1: #базовый случай - медиана и есть элемент
        return arr[0]

    pivot = random.choice(arr)

    # Разбиваем массив относительно pivot за O(n) на подмассивы
    lows = [x for x in arr if x < pivot]
    highs = [x for x in arr if x > pivot]
    pivots = [x for x in arr if x == pivot]

    if k < len(lows):
        # искомый элемент среди меньших pivot - ищем в lows
        return quickselect(lows, k)
    elif k < len(lows) + len(pivots):
        # искомый элемент равен pivot
        return pivots[0]
    else:
        # искомый элемент среди больших pivot - смещаем k и ищем в highs
        return quickselect(highs, k - len(lows) - len(pivots))


def median(data: list[float]) -> float:
    """
    Медиана data. Использует quickselect вместо сортировки всего списка.
    """
    validate(data)
    n = len(data)
    if n % 2 != 0:
        # Нечётная длина - медиана это средний элемент по счёту (индекс n//2)
        return quickselect(data, n // 2)
    else:
        # Для чётной длины — среднее двух центральных элементов
        left = quickselect(data, n // 2 - 1)
        right = quickselect(data, n // 2)
        return (left + right) / 2


def variance(data: list[float]) -> float:
    """
    Дисперсия data 
    """
    validate(data)
    m = mean(data)
    # дисперсия = сумма квадратов отклонений от среднего, делённая на n
    return sum((x - m) ** 2 for x in data) / len(data)


def std(data: list[float]) -> float:
    """
    Стандартное отклонение data (квадратный корень из variance).
    """
    validate(data)
    # стандартное отклонение - просто корень из дисперсии, чтобы вернуть
    # разброс в тех же единицах, что и сами данные (variance даёт квадрат единиц)
    return variance(data) ** 0.5


def min_max_scale(data: list[float], feature_range: tuple[float, float] = (0, 1)) -> list[float]:
    """
    Приводит значения data к диапазону feature_range.
    """
    validate(data)
    low, high = feature_range
    if low >= high:
        raise ValueError("feature_range: нижняя граница должна быть меньше верхней")
    data_min = min(data)
    data_max = max(data)
    if data_min == data_max:
        # все значения одинаковые - формула ниже делила бы на 0
        raise ValueError("нельзя масштабировать данные с одинаковыми значениями")
    return [
        # (x - data_min) / (data_max - data_min) сначала нормирует x в [0, 1],
        # low + ... * (high - low) растягивает этот [0, 1] в нужный диапазон
        low + (x - data_min) * (high - low) / (data_max - data_min)
        for x in data
    ]


def log_scale(data: list[float], base: float = math.e) -> list[float]:
    """
    Приводит значения data к логарифмической шкале по основанию base.
    """
    validate(data)
    if any(x <= 0 for x in data):
        # логарифм не определён для нуля и отрицательных чисел
        raise ValueError("log_scale применим только к положительным значениям")
    # логарифмируем каждое значение по заданному основанию
    return [math.log(x, base) for x in data]


def standardize(data: list[float]) -> list[float]:
    """
    Приводит data к нулевому среднему и единичному стандартному отклонению (z-score).
    """
    validate(data)
    m = mean(data)
    s = std(data)
    if s == 0:
        # все значения одинаковые - std=0, делить не на что
        raise ValueError("нельзя стандартизировать данные с нулевым стандартным отклонением")
    # z-score: сколько стандартных отклонений значение отстоит от среднего
    return [(x - m) / s for x in data]


def remove_outliers(data: list[float], k: float) -> list[float]:
    """
    Удаляет из data значения, отличающиеся от среднего более чем на k стандартных отклонений.
    """
    validate(data)
    if not isinstance(k, (int, float)) or k <= 0:
        raise ValueError("k должен быть положительным числом")
    m = mean(data)
    s = std(data)
    # правило "k сигм": оставляем только значения, отклонение которых
    # от среднего (в стандартных отклонениях) не превышает k
    return [x for x in data if abs(x - m) <= k * s]


def apply_pipeline(data: list[float], transformations: list[callable]) -> list[float]:
    """
    Последовательно применяет функции из transformations к набору данных
    """
    l = data
    for func in transformations:
        try:
            # результат текущего шага становится входом для следующего -
            # именно поэтому это конвейер, а не набор независимых вызовов
            l = func(l)
        except (TypeError, ValueError) as e:
            # если шаг сломался - сообщаем об этом и продолжаем со старым l,
            # не прерывая весь конвейер целиком
            print(f'В функции {func} ошибка: {e}')
    return l