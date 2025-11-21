import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from airlines.pipeline import DataProcessingPipeline
from airlines.analysis import DataAnalyzer
from airlines.visualization import DataVisualizer

def create_sample_data():
    """Создает тестовый CSV файл если его нет"""
    if not os.path.exists("airlines/airlines.csv"):
        print("Создание тестового airlines.csv...")
        
        # Генерируем тестовые данные
        airports = ['ATL', 'LAX', 'ORD', 'DFW', 'DEN', 'JFK', 'SFO', 'SEA', 'MCO', 'LAS']
        years = [2018, 2019, 2020, 2021, 2022]
        
        data = []
        for year in years:
            for airport in airports:
                for month in range(1, 13):
                    total_flights = np.random.randint(500, 5000)
                    delayed = np.random.randint(50, total_flights // 3)
                    cancelled = np.random.randint(0, total_flights // 20)
                    
                    data.append({
                        'Airport.Code': airport,
                        'Airport.Name': f'Airport {airport}',
                        'Time.Year': year,
                        'Time.Month': month,
                        'Time.Month Name': 'January',
                        'Statistics.Flights.Total': total_flights,
                        'Statistics.Flights.Delayed': delayed,
                        'Statistics.Flights.Cancelled': cancelled,
                        'Statistics.Flights.On Time': total_flights - delayed - cancelled
                    })
        
        df = pd.DataFrame(data)
        df.to_csv("airlines/airlines.csv", index=False)
        print(f"Создан файл с {len(df)} строками")

def run_task1():
    """Задание 1: Лучшие и худшие месяцы"""
    print("\n" + "="*60)
    print("ЗАДАНИЕ 1: ЛУЧШИЕ И ХУДШИЕ МЕСЯЦЫ ПО ПРОБЛЕМНЫМ РЕЙСАМ")
    print("="*60)
    
    analyzer = DataAnalyzer()
    visualizer = DataVisualizer(show_plots=DataVisualizer.show_plots)
    
    print("Анализ месяцев по доле задержанных/отмененных рейсов...")
    monthly_stats = analyzer.aggregate_best_worst_months("airlines/airlines.csv")
    
    if not monthly_stats.empty:
        print("\nВсе месяцы (от лучших к худшим):")
        print(monthly_stats.to_string(index=False))
        
        print("\nТоп-3 лучших месяца (наименьший % проблем):")
        print(monthly_stats.head(3).to_string(index=False))
        
        print("\nТоп-3 худших месяца (наибольший % проблем):")
        print(monthly_stats.tail(3).to_string(index=False))
        
        print("\nСтроим график...")
        visualizer.plot_best_worst_months(monthly_stats)
    else:
        print("Нет данных для анализа месяцев")

def run_task2():
    """Задание 2: Аэропорты с разбросом проблемных рейсов"""
    print("\n" + "="*60)
    print("ЗАДАНИЕ 2: АЭРОПОРТЫ С РАЗБРОСОМ ПРОБЛЕМНЫХ РЕЙСОВ")
    print("="*60)
    
    analyzer = DataAnalyzer()
    visualizer = DataVisualizer(show_plots=DataVisualizer.show_plots)
    
    print("Анализ разброса проблемных рейсов по аэропортам...")
    airport_variance = analyzer.analyze_airport_variance("airlines/airlines.csv")
    
    if not airport_variance.empty:
        print("\nВсе аэропорты (от наименьшего к наибольшему разбросу):")
        print(airport_variance.to_string(index=False))
        
        print("\nТоп-3 аэропорта с наименьшим разбросом (стабильные):")
        print(airport_variance.head(3).to_string(index=False))
        
        print("\nТоп-3 аэропорта с наибольшим разбросом (нестабильные):")
        print(airport_variance.tail(3).to_string(index=False))
        
        print("\nСтроим график...")
        visualizer.plot_airport_variance(airport_variance)
    else:
        print("Нет данных для анализа дисперсии аэропортов")

def run_time_series_analysis():
    """Задание 3: Временные ряды и скользящее среднее"""
    print("\n" + "="*60)
    print("ЗАДАНИЕ 3: ВРЕМЕННЫЕ РЯДЫ И СКОЛЬЗЯЩЕЕ СРЕДНЕЕ")
    print("="*60)
    
    analyzer = DataAnalyzer()
    visualizer = DataVisualizer(show_plots=DataVisualizer.show_plots)
    
    print("Анализ временных рядов со скользящим средним...")
    time_series_data = analyzer.analyze_time_series_with_ma("airlines/airlines.csv")
    
    if not time_series_data.empty:
        print(f"\nПроанализировано {len(time_series_data)} временных точек")
        print("\nПоследние 5 значений:")
        print(time_series_data.tail().to_string(index=False))
        
        print("\nСтроим график временного ряда...")
        visualizer.plot_time_series(time_series_data)
    else:
        print("Нет данных для анализа временных рядов")

def run_task3():
    """Задание 3: Самый загруженный аэропорт"""
    print("\n" + "="*60)
    print("ЗАДАНИЕ 3: САМЫЙ ЗАГРУЖЕННЫЙ АЭРОПОРТ")
    print("="*60)
    
    analyzer = DataAnalyzer()
    visualizer = DataVisualizer(show_plots=DataVisualizer.show_plots)
    
    print("Поиск самого загруженного аэропорта...")
    busiest_data = analyzer.analyze_busiest_airport("airlines/airlines.csv")
    
    if busiest_data:
        print(f"\nРЕЗУЛЬТАТЫ:")
        print(f"Самый загруженный аэропорт: {busiest_data['airport_code']}")
        print(f"Общее количество рейсов: {busiest_data['total_flights']:,}")
        
        print(f"\nТоп-5 самых загруженных аэропортов:")
        top_airports = sorted(busiest_data['all_airports'].items(), 
                            key=lambda x: x[1], reverse=True)[:5]
        for i, (airport, flights) in enumerate(top_airports, 1):
            print(f"{i}. {airport}: {flights:,} рейсов")
        
        print("\nСтроим график...")
        visualizer.plot_busiest_airports(busiest_data)
    else:
        print("Нет данных для анализа загруженности аэропортов")
    
    run_time_series_analysis()

def compare_performance(csv_file: str, parquet_file: str) -> Dict:
    """Сравнение производительности CSV и Parquet с возвратом данных для графика"""
    pipeline = DataProcessingPipeline()
    visualizer = DataVisualizer(show_plots=DataVisualizer.show_plots)
    
    performance_data = {}
    
    # Конвертация в Parquet если нужно
    try:
        if not os.path.exists(parquet_file):
            print("Создание Parquet файла...")
            pipeline.csv_to_parquet(csv_file, parquet_file)
    except Exception as e:
        print(f"Ошибка при создании Parquet файла: {e}")
        return performance_data
    
    # Тестирование CSV
    print("Тестирование чтения CSV...")
    start_time = time.time()
    try:
        csv_data = pd.read_csv(csv_file)
        csv_time = time.time() - start_time
        performance_data['csv_time'] = csv_time
        print(f"CSV: Прочитано {len(csv_data)} строк, {len(csv_data.columns)} столбцов")
        print(f"CSV время: {csv_time:.6f} сек")
    except Exception as e:
        print(f"Ошибка чтения CSV: {e}")
        csv_time = float('inf')
    
    # Тестирование Parquet (полное чтение)
    print("Тестирование чтения Parquet (все столбцы)...")
    start_time = time.time()
    try:
        parquet_data_full = pd.read_parquet(parquet_file)
        parquet_full_time = time.time() - start_time
        performance_data['parquet_full_time'] = parquet_full_time
        print(f"Parquet (все): Прочитано {len(parquet_data_full)} строк, {len(parquet_data_full.columns)} столбцов")
        print(f"Parquet (все) время: {parquet_full_time:.6f} сек")
    except Exception as e:
        print(f"Ошибка чтения Parquet (все): {e}")
        parquet_full_time = float('inf')
    
    # Тестирование Parquet (выборочное чтение)
    print("Тестирование чтения Parquet (выборочные столбцы)...")
    start_time = time.time()
    try:
        parquet_data_partial = pd.read_parquet(parquet_file, columns=[
            'Airport.Code', 
            'Airport.Name',
            'Time.Year',
            'Time.Month',
            'Statistics.Flights.Delayed',
            'Statistics.Flights.Total'
        ])
        parquet_partial_time = time.time() - start_time
        performance_data['parquet_partial_time'] = parquet_partial_time
        print(f"Parquet (выборочно): Прочитано {len(parquet_data_partial)} строк, {len(parquet_data_partial.columns)} столбцов")
        print(f"Parquet (выборочно) время: {parquet_partial_time:.6f} сек")
    except Exception as e:
        print(f"Ошибка чтения Parquet (выборочно): {e}")
        parquet_partial_time = float('inf')
    
    # Сравнение результатов
    print("\n" + "-" * 40)
    print("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ:")
    print("-" * 40)
    
    # ИСПРАВЛЕНИЕ: Добавляем проверку на нулевое время
    if csv_time < float('inf') and parquet_full_time > 0 and parquet_full_time < float('inf'):
        speedup_full = csv_time / parquet_full_time
        performance_data['speedup_full'] = speedup_full
        print(f"Parquet (все) ускоряет чтение в {speedup_full:.2f} раз")
    else:
        print("Невозможно рассчитать ускорение для полного чтения Parquet")
    
    # ИСПРАВЛЕНИЕ: Добавляем проверку на нулевое время
    if csv_time < float('inf') and parquet_partial_time > 0 and parquet_partial_time < float('inf'):
        speedup_partial = csv_time / parquet_partial_time
        performance_data['speedup_partial'] = speedup_partial
        print(f"Parquet (выборочно) ускоряет чтение в {speedup_partial:.2f} раз")
    else:
        print("Невозможно рассчитать ускорение для выборочного чтения Parquet")
    
    # Если время равно 0, устанавливаем минимальное значение для визуализации
    if performance_data.get('parquet_full_time', 0) == 0:
        performance_data['parquet_full_time'] = 0.000001
    if performance_data.get('parquet_partial_time', 0) == 0:
        performance_data['parquet_partial_time'] = 0.000001
    
    # Строим график сравнения производительности
    print("\nСтроим график сравнения производительности...")
    visualizer.plot_performance_comparison(performance_data)
    
    return performance_data

def run_additional_task():
    """Дополнительное задание: работа с Parquet"""
    print("\n" + "="*60)
    print("ДОПОЛНИТЕЛЬНОЕ ЗАДАНИЕ: PARQUET ОПТИМИЗАЦИЯ")
    print("="*60)
    
    pipeline = DataProcessingPipeline()
    analyzer = DataAnalyzer()
    visualizer = DataVisualizer(show_plots=DataVisualizer.show_plots)
    
    csv_file = "airlines/airlines.csv"
    parquet_file = "airlines/airlines.parquet"
    
    # Сравнение производительности
    print("1. Сравнение скорости чтения CSV и Parquet:")
    compare_performance(csv_file, parquet_file)
    
    # Анализ данных из Parquet
    print("\n2. Анализ данных аэропортов из Parquet файла:")
    airport_stats = analyzer.get_airport_delay_stats(parquet_file)
    
    if not airport_stats.empty:
        print(f"\nПроанализировано {len(airport_stats)} аэропортов")
        print("\nТоп-10 аэропортов по проценту задержек:")
        print(airport_stats.head(10).to_string(index=False))
        
        print("\nСтроим scatter plot...")
        visualizer.plot_airport_delays(airport_stats, top_n=15)
    else:
        print("Нет данных для анализа аэропортов")

def run_correlation_analysis():
    """Дополнительное задание: Анализ корреляции с использованием Parquet"""
    print("\n" + "="*60)
    print("ДОПОЛНИТЕЛЬНОЕ ЗАДАНИЕ: АНАЛИЗ КОРРЕЛЯЦИИ (PARQUET)")
    print("="*60)
    
    analyzer = DataAnalyzer()
    visualizer = DataVisualizer(show_plots=DataVisualizer.show_plots)
    
    parquet_file = "airlines/airlines.parquet"
    
    print("Анализ корреляции с использованием Parquet (выборочное чтение столбцов)...")
    
    # Получаем данные для корреляции ИЗ PARQUET
    correlation_result = analyzer.analyze_correlation_with_parquet(parquet_file)
    
    if correlation_result:
        print(f"\nРЕЗУЛЬТАТЫ КОРРЕЛЯЦИОННОГО АНАЛИЗА (Parquet):")
        print(f"Количество образцов: {correlation_result['n_samples']}")
        print(f"Корреляция (Задержки vs Всего): {correlation_result['correlation_delayed']:.4f}")
        print(f"Корреляция (Отмены vs Всего): {correlation_result['correlation_cancelled']:.4f}")
        print(f"Корреляция (По расписанию vs Всего): {correlation_result['correlation_on_time']:.4f}")
        
        # Интерпретация результатов
        print(f"\nИНТЕРПРЕТАЦИЯ:")
        delayed_corr = correlation_result['correlation_delayed']
        cancelled_corr = correlation_result['correlation_cancelled'] 
        on_time_corr = correlation_result['correlation_on_time']
        
        print(f"Задержки: {visualizer.get_correlation_interpretation(delayed_corr)} связь")
        print(f"Отмены: {visualizer.get_correlation_interpretation(cancelled_corr)} связь")
        print(f"По расписанию: {visualizer.get_correlation_interpretation(on_time_corr)} связь")
        
        # Выводы
        print(f"\nВЫВОДЫ:")
        if delayed_corr > 0.7:
            print("• Сильная положительная корреляция: больше рейсов → больше задержек")
        elif delayed_corr > 0.3:
            print("• Умеренная положительная корреляция: рост рейсов связан с ростом задержек") 
        else:
            print("• Слабая корреляция: количество рейсов слабо влияет на задержки")
        
        # Используем данные из correlation_result (уже прочитаны из Parquet)
        print("\nСтроим scatter plot корреляции...")
        visualizer.plot_correlation_analysis(correlation_result, correlation_result['raw_data'])
    else:
        print("Не удалось выполнить анализ корреляции с Parquet")
        
def main_lab3(show_plots: bool = False):
    """Основная функция лабораторной работы 3"""
    print("ЛАБОРАТОРНАЯ РАБОТА 3: АНАЛИЗ ДАННЫХ АЭРОПОРТОВ")
    print("=" * 60)
    print(f"РЕЖИМ: {'ПОКАЗЫВАТЬ ГРАФИКИ' if show_plots else 'ТОЛЬКО СОХРАНЕНИЕ'}")
    print("=" * 60)
    
    # Передаем параметр show_plots в визуализатор
    from airlines.visualization import DataVisualizer
    DataVisualizer.show_plots = show_plots
    
    # Создаем тестовые данные если файла нет
    if not os.path.exists("airlines/airlines.csv"):
        create_sample_data()
    
    # Проверяем наличие файла данных
    if not os.path.exists("airlines/airlines.csv"):
        print("ОШИБКА: Файл 'airlines/airlines.csv' не найден!")
        return
    
    try:
        # Выполняем все задания
        run_task1()
        run_task2() 
        run_task3()
        run_additional_task()
        run_correlation_analysis()
        
        print("\n" + "=" * 60)
        print("ЛАБОРАТОРНАЯ РАБОТА 3 УСПЕШНО ЗАВЕРШЕНА!")
        if not show_plots:
            print("Графики сохранены в папку 'plots'")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nОШИБКА при выполнении лабораторной работы: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_lab3()