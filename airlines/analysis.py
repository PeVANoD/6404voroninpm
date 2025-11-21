import pandas as pd
import numpy as np
import os
from scipy import stats
from typing import Dict, List, Tuple
from airlines.pipeline import DataProcessingPipeline

class DataAnalyzer:
    def __init__(self):
        self.pipeline = DataProcessingPipeline()
    
    def aggregate_yearly_stats(self, file_path: str) -> pd.DataFrame:
        """Задание 1: Агрегация данных по годам"""
        chunks = self.pipeline.read_csv_chunks(file_path)
        chunks = self.pipeline.clean_data(chunks)
        
        agg_columns = {
            'Statistics.Flights.Total': 'sum',
            'Statistics.Flights.Delayed': 'sum',
            'Statistics.Flights.Cancelled': 'sum',
            'Statistics.Flights.On Time': 'sum'
        }
        
        return self.pipeline.aggregate_chunks(chunks, 'Time.Year', agg_columns)
    
    def calculate_confidence_intervals(self, file_path: str) -> Dict[str, List]:
        """Задание 2: Дисперсия и доверительные интервалы"""
        delays_data = []
        
        chunks = self.pipeline.read_csv_chunks(file_path)
        chunks = self.pipeline.clean_data(chunks)
        
        for chunk in chunks:
            if 'Statistics.Flights.Delayed' in chunk.columns and 'Statistics.Flights.Total' in chunk.columns:
                # Рассчитываем процент задержанных рейсов
                chunk = chunk.copy()
                chunk['Delay_Rate'] = (chunk['Statistics.Flights.Delayed'] / 
                                     chunk['Statistics.Flights.Total']) * 100
                delays_data.extend(chunk['Delay_Rate'].dropna().tolist())
        
        if not delays_data:
            return {}
        
        delays_array = np.array(delays_data)
        
        # Расчет статистик
        mean = np.mean(delays_array)
        std = np.std(delays_array, ddof=1)
        n = len(delays_array)
        
        # 95% доверительный интервал
        confidence = 0.95
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin_of_error = t_value * (std / np.sqrt(n))
        
        return {
            'mean': mean,
            'std': std,
            'confidence_interval': [mean - margin_of_error, mean + margin_of_error],
            'n': n
        }
    
    def analyze_time_series(self, file_path: str) -> pd.DataFrame:
        """Задание 3: Временные ряды и скользящее среднее"""
        all_data = []
        
        chunks = self.pipeline.read_csv_chunks(file_path)
        chunks = self.pipeline.clean_data(chunks)
        
        for chunk in chunks:
            required_cols = ['Time.Year', 'Time.Month', 'Statistics.Flights.Delayed', 'Statistics.Flights.Total']
            if all(col in chunk.columns for col in required_cols):
                chunk = chunk.copy()
                chunk['Date'] = pd.to_datetime(
                    chunk['Time.Year'].astype(str) + '-' + chunk['Time.Month'].astype(str) + '-01'
                )
                chunk['Delay_Rate'] = (chunk['Statistics.Flights.Delayed'] / 
                                    chunk['Statistics.Flights.Total']) * 100
                all_data.append(chunk[['Date', 'Delay_Rate']])
        
        if all_data:
            time_series = pd.concat(all_data, ignore_index=True)
            time_series = time_series.sort_values('Date')
            
            # Скользящее среднее за 3 месяца - ИСПРАВЛЕННОЕ название
            time_series['Moving_Avg_3'] = time_series['Delay_Rate'].rolling(
                window=3, min_periods=1
            ).mean()
            
            return time_series
        
        return pd.DataFrame()
    
    def get_airport_delay_stats(self, parquet_file: str) -> pd.DataFrame:
        """Дополнительное задание: статистика задержек по аэропортам из Parquet"""
        try:
            # Читаем только нужные столбцы из Parquet
            data = pd.read_parquet(parquet_file, columns=[
                'Airport.Code', 
                'Airport.Name',
                'Statistics.Flights.Delayed',
                'Statistics.Flights.Total'
            ])
            
            # Агрегируем по аэропортам
            airport_stats = data.groupby(['Airport.Code', 'Airport.Name']).agg({
                'Statistics.Flights.Delayed': 'sum',
                'Statistics.Flights.Total': 'sum'
            }).reset_index()
            
            # Рассчитываем процент задержек
            airport_stats['Delay_Percentage'] = (
                airport_stats['Statistics.Flights.Delayed'] / 
                airport_stats['Statistics.Flights.Total']
            ) * 100
            
            return airport_stats.sort_values('Delay_Percentage', ascending=False)
            
        except Exception as e:
            print(f"Ошибка при чтении Parquet: {e}")
            return pd.DataFrame()
    
    def analyze_correlation_with_parquet(self, parquet_file: str) -> Dict:
        """Доп. задание с использованием Parquet: Корреляционный анализ с выборочным чтением столбцов"""
        print("Анализ корреляции с использованием Parquet (выборочное чтение столбцов)...")
        
        try:
            # Читаем ТОЛЬКО нужные столбцы из Parquet (без генератора)
            data = pd.read_parquet(parquet_file, columns=[
                'Statistics.Flights.Total',
                'Statistics.Flights.Delayed', 
                'Statistics.Flights.Cancelled',
                'Statistics.Flights.On Time'
            ])
            
            # Удаляем строки с пропущенными значениями
            data_clean = data.dropna()
            
            if len(data_clean) < 2:
                return {}
            
            # Рассчитываем корреляции
            correlation_delayed = data_clean['Statistics.Flights.Delayed'].corr(
                data_clean['Statistics.Flights.Total']
            )
            
            correlation_cancelled = data_clean['Statistics.Flights.Cancelled'].corr(
                data_clean['Statistics.Flights.Total']
            )
            
            correlation_on_time = data_clean['Statistics.Flights.On Time'].corr(
                data_clean['Statistics.Flights.Total']
            )
            
            return {
                'correlation_delayed': correlation_delayed,
                'correlation_cancelled': correlation_cancelled, 
                'correlation_on_time': correlation_on_time,
                'n_samples': len(data_clean),
                'raw_data': data_clean  # Возвращаем данные для scatter plot
            }
            
        except Exception as e:
            print(f"Ошибка анализа корреляции с Parquet: {e}")
            return {}
        
    def aggregate_best_worst_months(self, file_path: str) -> pd.DataFrame:
        """Задание 1: 3 лучших и 3 худших месяца по доле задержанных/отмененных рейсов"""
        chunks = self.pipeline.read_csv_chunks(file_path)
        chunks = self.pipeline.clean_data(chunks)
        
        # Используем инкрементальную агрегацию
        monthly_stats = self.pipeline.aggregate_chunks_incremental(chunks, 
            ['Time.Year', 'Time.Month', 'Time.Month Name'], 
            {
                'Statistics.Flights.Delayed': 'sum',
                'Statistics.Flights.Cancelled': 'sum', 
                'Statistics.Flights.Total': 'sum'
            }
        )
        
        if not monthly_stats.empty:
            # Рассчитываем долю проблемных рейсов
            monthly_stats['Problem_Rate'] = (
                (monthly_stats['Statistics.Flights.Delayed'] + monthly_stats['Statistics.Flights.Cancelled']) / 
                monthly_stats['Statistics.Flights.Total']
            ) * 100
            
            # Группируем по месяцам (усредняем по годам)
            final_monthly = monthly_stats.groupby(['Time.Month', 'Time.Month Name']).agg({
                'Problem_Rate': 'mean',
                'Statistics.Flights.Total': 'sum'
            }).reset_index()
            
            # Сортируем по Problem_Rate (лучшие - наименьший процент проблем)
            final_monthly = final_monthly.sort_values('Problem_Rate')
            
            return final_monthly
        
        return pd.DataFrame()

    def analyze_airport_variance(self, file_path: str) -> pd.DataFrame:
        """Задание 2: Аэропорты с наибольшим и наименьшим разбросом задержанных/отмененных рейсов"""
        # Используем Parquet для эффективного чтения только нужных столбцов
        parquet_file = file_path.replace('.csv', '.parquet')
        
        # Создаем Parquet файл если нужно
        if not os.path.exists(parquet_file):
            self.pipeline.csv_to_parquet(file_path, parquet_file)
        
        try:
            # Читаем только нужные столбцы
            data = pd.read_parquet(parquet_file, columns=[
                'Airport.Code', 
                'Airport.Name',
                'Statistics.Flights.Delayed', 
                'Statistics.Flights.Cancelled',
                'Statistics.Flights.Total'
            ])
            
            # Рассчитываем процент проблемных рейсов для каждого наблюдения
            data['Problem_Rate'] = (
                (data['Statistics.Flights.Delayed'] + data['Statistics.Flights.Cancelled']) / 
                data['Statistics.Flights.Total']
            ) * 100
            
            # Вычисляем стандартное отклонение (разброс) по аэропортам
            airport_variance = data.groupby(['Airport.Code', 'Airport.Name']).agg({
                'Problem_Rate': 'std',
                'Statistics.Flights.Total': 'sum'
            }).reset_index()
            
            airport_variance = airport_variance.sort_values('Problem_Rate')
            return airport_variance
            
        except Exception as e:
            print(f"Ошибка анализа дисперсии аэропортов: {e}")
            return pd.DataFrame()

    def analyze_busiest_airport(self, file_path: str) -> Dict:
        """Задание 3: Самый загруженный аэропорт за весь период"""
        chunks = self.pipeline.read_csv_chunks(file_path)
        chunks = self.pipeline.clean_data(chunks)
        
        # Используем инкрементальную агрегацию
        airport_aggregates = self.pipeline.aggregate_chunks_incremental(chunks,
            'Airport.Code',
            {'Statistics.Flights.Total': 'sum'}
        )
        
        if not airport_aggregates.empty:
            # Создаем словарь с результатами
            airport_totals = dict(zip(
                airport_aggregates['Airport.Code'], 
                airport_aggregates['Statistics.Flights.Total']
            ))
            
            # Находим самый загруженный аэропорт
            busiest_airport = max(airport_totals.items(), key=lambda x: x[1])
            
            return {
                'airport_code': busiest_airport[0],
                'total_flights': busiest_airport[1],
                'all_airports': airport_totals
            }
        
        return {}
    
    def analyze_time_series_with_ma(self, file_path: str, window: int = 8) -> pd.DataFrame:
        """Задание 3: Временные ряды и скользящее среднее"""
        all_data = []
        
        chunks = self.pipeline.read_csv_chunks(file_path)
        chunks = self.pipeline.clean_data(chunks)
        
        for chunk in chunks:
            required_cols = ['Time.Year', 'Time.Month', 'Statistics.Flights.Delayed', 'Statistics.Flights.Total']
            if all(col in chunk.columns for col in required_cols):
                chunk = chunk.copy()
                chunk['Date'] = pd.to_datetime(
                    chunk['Time.Year'].astype(str) + '-' + chunk['Time.Month'].astype(str) + '-01'
                )
                chunk['Delay_Rate'] = (chunk['Statistics.Flights.Delayed'] / 
                                    chunk['Statistics.Flights.Total']) * 100
                all_data.append(chunk[['Date', 'Delay_Rate']])
        
        if all_data:
            time_series = pd.concat(all_data, ignore_index=True)
            
            # ВАЖНО: сортируем по дате и убираем дубликаты
            time_series = time_series.sort_values('Date').drop_duplicates('Date')
            
            # Скользящее среднее - УЛУЧШЕННАЯ версия
            time_series['Moving_Avg_3'] = time_series['Delay_Rate'].rolling(
                window=window, min_periods=1
            ).mean()
            
            return time_series
        
        return pd.DataFrame()