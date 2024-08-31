import numpy as np
import pandas as pd

def find_outliers_iqr(data, feature, left=1.5, right=1.5, log_scale=False):
    """
    Находит выбросы в данных, используя метод межквартильного размаха. 
    Классический метод модифицирован путем добавления:
    * возможности логарифмирования распредления
    * ручного управления количеством межквартильных размахов в обе стороны распределения
    Args:
        data (pandas.DataFrame): набор данных
        feature (str): имя признака, на основе которого происходит поиск выбросов
        left (float, optional): количество межквартильных размахов в левую сторону распределения. По умолчанию 1.5.
        right (float, optional): количество межквартильных размахов в правую сторону распределения. По умолчанию 1.5.
        log_scale (bool, optional): режим логарифмирования. По умолчанию False - логарифмирование не применяется.

    Returns:
        pandas.DataFrame: наблюдения, попавшие в разряд выбросов
        pandas.DataFrame: очищенные данные, из которых исключены выбросы
    """
    if log_scale:
        x = np.log(data[feature]+1)
    else:
        x= data[feature]
    quartile_1, quartile_3 = x.quantile(0.25), x.quantile(0.75),
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * left)
    upper_bound = quartile_3 + (iqr * right)
    outliers = data[(x<lower_bound) | (x > upper_bound)]
    cleaned = data[(x>lower_bound) & (x < upper_bound)]
    return outliers, cleaned

def find_outliers_z_score(data, feature, left=3, right=3, log_scale=False):
    """
    Находит выбросы в данных, используя метод z-отклонений. 
    Классический метод модифицирован путем добавления:
    * возможности логарифмирования распредления
    * ручного управления количеством стандартных отклонений в обе стороны распределения
    Args:
        data (pandas.DataFrame): набор данных
        feature (str): имя признака, на основе которого происходит поиск выбросов
        left (float, optional): количество стандартных отклонений в левую сторону распределения. По умолчанию 1.5.
        right (float, optional): количество стандартных в правую сторону распределения. По умолчанию 1.5.
        log_scale (bool, optional): режим логарифмирования. По умолчанию False - логарифмирование не применяется.

    Returns:
        pandas.DataFrame: наблюдения, попавшие в разряд выбросов
        pandas.DataFrame: очищенные данные, из которых исключены выбросы
    """
    if log_scale:
        x = np.log(data[feature]+1)
    else:
        x = data[feature]
    mu = x.mean()
    sigma = x.std()
    lower_bound = mu - left * sigma
    upper_bound = mu + right * sigma
    outliers = data[(x < lower_bound) | (x > upper_bound)]
    cleaned = data[(x > lower_bound) & (x < upper_bound)]
    return outliers, cleaned


def find_outliers_quantile(data, feature, left=0.01, right=0.99):
    """
    Функция для поиска выбросов в данных на основе квантилей.

    Параметры:
    ----------
    data : pandas.DataFrame
        Входные данные в виде DataFrame, содержащие признак для анализа.
        
    feature : str
        Имя столбца в DataFrame, по которому необходимо найти выбросы.
        
    left : float, optional, default=0.01
        Левый квантиль для определения нижней границы выбросов.
        
    right : float, optional, default=0.99
        Правый квантиль для определения верхней границы выбросов.

    Возвращаемые значения:
    ----------------------
    outliers : pandas.DataFrame
        DataFrame, содержащий строки с выбросами, которые находятся за пределами 
        указанных квантилей.
        
    cleaned : pandas.DataFrame
        DataFrame, содержащий строки без выбросов, находящиеся между указанными квантилями.

    Пример использования:
    ---------------------
    outliers, cleaned = find_outliers_quantile(data, 'price', left=0.05, right=0.95)
    """
    x = data[feature]
    lower_bound = x.quantile(left)
    upper_bound = x.quantile(right)
    outliers = data[(x < lower_bound) | (x > upper_bound)]
    cleaned = data[(x > lower_bound) & (x < upper_bound)]
    return outliers, cleaned




