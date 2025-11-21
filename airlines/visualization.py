import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from typing import Dict, List

class DataVisualizer:
    def __init__(self, save_plots: bool = True, show_plots: bool = False):
        plt.style.use('seaborn-v0_8')
        self.fig_size = (12, 8)
        self.save_plots = save_plots
        self.show_plots = show_plots
        self.plot_dir = "plots"
        
        # Создаем папку для графиков если нужно
        if self.save_plots and not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
    
    def _save_plot(self, filename: str):
        """Сохраняет текущий график в файл"""
        if self.save_plots:
            filepath = os.path.join(self.plot_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ График сохранен: {filepath}")
    
    def _display_plot(self):
        """Показывает график в зависимости от настроек"""
        if self.show_plots:
            plt.show()
        else:
            plt.close()

    def plot_yearly_stats(self, yearly_data: pd.DataFrame):
        """График 1: Агрегация данных по годам (сохранение + показ)"""
        if yearly_data.empty:
            print("Нет данных для построения графика")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.fig_size)
        fig.suptitle('Статистика рейсов по годам', fontsize=16, fontweight='bold')
        
        # Общее количество рейсов
        ax1.bar(yearly_data['Time.Year'], yearly_data['Statistics.Flights.Total'], 
                color='skyblue', alpha=0.7)
        ax1.set_title('Общее количество рейсов')
        ax1.set_xlabel('Год')
        ax1.set_ylabel('Количество рейсов')
        ax1.grid(True, alpha=0.3)
        
        # Задержанные рейсы
        ax2.bar(yearly_data['Time.Year'], yearly_data['Statistics.Flights.Delayed'],
                color='lightcoral', alpha=0.7)
        ax2.set_title('Задержанные рейсы')
        ax2.set_xlabel('Год')
        ax2.set_ylabel('Задержанные рейсы')
        ax2.grid(True, alpha=0.3)
        
        # Отмененные рейсы
        ax3.bar(yearly_data['Time.Year'], yearly_data['Statistics.Flights.Cancelled'],
                color='gold', alpha=0.7)
        ax3.set_title('Отмененные рейсы')
        ax3.set_xlabel('Год')
        ax3.set_ylabel('Отмененные рейсы')
        ax3.grid(True, alpha=0.3)
        
        # Рейсы по расписанию
        ax4.bar(yearly_data['Time.Year'], yearly_data['Statistics.Flights.On Time'],
                color='lightgreen', alpha=0.7)
        ax4.set_title('Рейсы по расписанию')
        ax4.set_xlabel('Год')
        ax4.set_ylabel('Рейсы по расписанию')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Сохраняем и показываем
        self._save_plot('1_yearly_stats.png')
        self._display_plot()
    
    def plot_confidence_intervals(self, confidence_data: Dict):
        """График 2: Дисперсия и доверительные интервалы (сохранение + показ)"""
        if not confidence_data:
            print("Нет данных для построения графика")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        mean = confidence_data['mean']
        ci_low, ci_high = confidence_data['confidence_interval']
        
        # Bar plot с доверительными интервалами
        bars = ax.bar(['Средний процент задержек'], [mean], 
                     yerr=[[mean - ci_low], [ci_high - mean]], 
                     capsize=15, alpha=0.7, color='lightseagreen',
                     edgecolor='darkgreen', linewidth=2)
        
        ax.set_ylabel('Процент задержек (%)', fontsize=12)
        ax.set_title('Доверительный интервал для процента задержек рейсов\n'
                    f'(95% ДИ: {ci_low:.2f}% - {ci_high:.2f}%, n={confidence_data["n"]})', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Добавляем аннотации
        ax.text(0, mean + (ci_high - mean) + 1, f'Среднее: {mean:.2f}%', 
               ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Добавляем линию для среднего
        ax.axhline(y=mean, color='red', linestyle='--', alpha=0.7, label=f'Среднее: {mean:.2f}%')
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Сохраняем и показываем
        self._save_plot('2_confidence_interval.png')
        self._display_plot()
    
    def plot_time_series(self, time_series_data: pd.DataFrame):
        """График 3: Временные ряды и скользящее среднее (сохранение + показ)"""
        if time_series_data.empty:
            print("Нет данных для построения графика")
            return
        
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Исходные данные
        ax.plot(time_series_data['Date'], time_series_data['Delay_Rate'], 
               alpha=0.6, label='Процент задержек', linewidth=1, color='blue',
               marker='o', markersize=3)
        
        # Скользящее среднее
        ax.plot(time_series_data['Date'], time_series_data['Moving_Avg_3'],
               linewidth=3, label='Скользящее среднее (3 месяца)', color='red',
               alpha=0.8)
        
        ax.set_title('Изменение процента задержек рейсов во времени', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Дата', fontsize=12)
        ax.set_ylabel('Процент задержек (%)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Форматирование дат
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Сохраняем и показываем
        self._save_plot('3_time_series.png')
        self._display_plot()
    
    def plot_airport_delays(self, airlines: pd.DataFrame, top_n: int = 15):
        """Дополнительный график: Задержки по аэропортам (Scatter plot)"""
        if airlines.empty:
            print("Нет данных для построения графика")
            return
        
        # Берем топ-N аэропортов
        top_airports = airlines.head(top_n)
        
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Scatter plot
        scatter = ax.scatter(top_airports['Statistics.Flights.Total'], 
                           top_airports['Delay_Percentage'],
                           s=100, alpha=0.7, 
                           c=top_airports['Delay_Percentage'],
                           cmap='viridis', edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Общее количество рейсов', fontsize=12)
        ax.set_ylabel('Процент задержек (%)', fontsize=12)
        ax.set_title(f'Топ-{top_n} аэропортов по проценту задержек\n(данные из Parquet файла)',
                    fontsize=14, fontweight='bold')
        
        # Добавляем цветовую шкалу
        cbar = plt.colorbar(scatter, label='Процент задержек (%)')
        cbar.ax.tick_params(labelsize=10)
        
        # Добавляем подписи для некоторых точек
        for i, row in top_airports.head(8).iterrows():
            ax.annotate(f"{row['Airport.Code']}\n({row['Delay_Percentage']:.1f}%)", 
                       (row['Statistics.Flights.Total'], row['Delay_Percentage']),
                       xytext=(8, 8), textcoords='offset points', 
                       fontsize=8, alpha=0.8,
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Сохраняем и показываем
        self._save_plot('4_airport_delays_scatter.png')
        self._display_plot()
    
    def plot_performance_comparison(self, performance_data: Dict):
        """График сравнения производительности CSV vs Parquet"""
        if not performance_data:
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ['CSV', 'Parquet (все)', 'Parquet (выборочно)']
        times = [
            performance_data.get('csv_time', 0),
            performance_data.get('parquet_full_time', 0),
            performance_data.get('parquet_partial_time', 0)
        ]
        
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        bars = ax.bar(methods, times, color=colors, alpha=0.7, edgecolor='black')
        
        # Добавляем значения на столбцы
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{time:.4f} сек', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Время чтения (секунды)', fontsize=12)
        ax.set_title('Сравнение скорости чтения: CSV vs Parquet', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Добавляем информацию об ускорении
        if performance_data.get('speedup_full'):
            ax.text(0.5, 0.95, f"Ускорение Parquet (все): {performance_data['speedup_full']:.2f}x",
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
        
        if performance_data.get('speedup_partial'):
            ax.text(0.5, 0.85, f"Ускорение Parquet (выборочно): {performance_data['speedup_partial']:.2f}x",
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.7))
        
        plt.tight_layout()
        self._save_plot('5_performance_comparison.png')
        self._display_plot()

    def plot_correlation_analysis(self, correlation_data: Dict, raw_data: pd.DataFrame = None):
        """График для анализа корреляции (Scatter plots)"""
        if not correlation_data:
            print("Нет данных для построения графика корреляции")
            return
        
        # Создаем сетку графиков
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Анализ корреляции: Задержки/Отмены vs Общее количество рейсов', 
                    fontsize=16, fontweight='bold')
        
        if raw_data is not None and not raw_data.empty:
            # Scatter plot: Задержанные рейсы vs Общее количество
            ax1.scatter(raw_data['Statistics.Flights.Total'], 
                    raw_data['Statistics.Flights.Delayed'],
                    alpha=0.6, color='red', s=30)
            ax1.set_xlabel('Общее количество рейсов')
            ax1.set_ylabel('Задержанные рейсы')
            ax1.set_title(f'Задержанные рейсы\nкорреляция: {correlation_data.get("correlation_delayed", 0):.3f}')
            ax1.grid(True, alpha=0.3)
            
            # Линия тренда для задержанных рейсов
            if len(raw_data) > 1:
                z = np.polyfit(raw_data['Statistics.Flights.Total'], 
                            raw_data['Statistics.Flights.Delayed'], 1)
                p = np.poly1d(z)
                ax1.plot(raw_data['Statistics.Flights.Total'], 
                        p(raw_data['Statistics.Flights.Total']), 
                        "r--", alpha=0.8, linewidth=2)
            
            # Scatter plot: Отмененные рейсы vs Общее количество
            ax2.scatter(raw_data['Statistics.Flights.Total'], 
                    raw_data['Statistics.Flights.Cancelled'],
                    alpha=0.6, color='orange', s=30)
            ax2.set_xlabel('Общее количество рейсов')
            ax2.set_ylabel('Отмененные рейсы')
            ax2.set_title(f'Отмененные рейсы\nкорреляция: {correlation_data.get("correlation_cancelled", 0):.3f}')
            ax2.grid(True, alpha=0.3)
            
            # Линия тренда для отмененных рейсов
            if len(raw_data) > 1:
                z = np.polyfit(raw_data['Statistics.Flights.Total'], 
                            raw_data['Statistics.Flights.Cancelled'], 1)
                p = np.poly1d(z)
                ax2.plot(raw_data['Statistics.Flights.Total'], 
                        p(raw_data['Statistics.Flights.Total']), 
                        "r--", alpha=0.8, linewidth=2)
            
            # Scatter plot: Рейсы по расписанию vs Общее количество
            ax3.scatter(raw_data['Statistics.Flights.Total'], 
                    raw_data['Statistics.Flights.On Time'],
                    alpha=0.6, color='green', s=30)
            ax3.set_xlabel('Общее количество рейсов')
            ax3.set_ylabel('Рейсы по расписанию')
            ax3.set_title(f'Рейсы по расписанию\nкорреляция: {correlation_data.get("correlation_on_time", 0):.3f}')
            ax3.grid(True, alpha=0.3)
            
            # Линия тренда для рейсов по расписанию
            if len(raw_data) > 1:
                z = np.polyfit(raw_data['Statistics.Flights.Total'], 
                            raw_data['Statistics.Flights.On Time'], 1)
                p = np.poly1d(z)
                ax3.plot(raw_data['Statistics.Flights.Total'], 
                        p(raw_data['Statistics.Flights.Total']), 
                        "r--", alpha=0.8, linewidth=2)
        
        # Bar plot: Сводка корреляций
        correlations = [
            correlation_data.get('correlation_delayed', 0),
            correlation_data.get('correlation_cancelled', 0),
            correlation_data.get('correlation_on_time', 0)
        ]
        labels = ['Задержанные', 'Отмененные', 'По расписанию']
        colors = ['red', 'orange', 'green']
        
        bars = ax4.bar(labels, correlations, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Коэффициент корреляции')
        ax4.set_title('Сводка корреляций')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Добавляем значения на столбцы
        for bar, correlation in zip(bars, correlations):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{correlation:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Добавляем общую информацию
        fig.text(0.5, 0.01, 
                f"Проанализировано образцов: {correlation_data.get('n_samples', 0)} | "
                f"Сильная корреляция: >0.7 | Умеренная: 0.3-0.7 | Слабая: <0.3",
                ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        # Сохраняем и показываем
        self._save_plot('6_correlation_analysis.png')
        self._display_plot()

    def plot_best_worst_months(self, monthly_data: pd.DataFrame):
        """График для задания 1: Лучшие и худшие месяцы"""
        if monthly_data.empty:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Лучшие и худшие месяцы по проценту проблемных рейсов', fontsize=16, fontweight='bold')
        
        # Лучшие месяцы
        best_months = monthly_data.head(3)
        bars1 = ax1.bar(range(len(best_months)), best_months['Problem_Rate'], 
                    color=['green', 'lightgreen', 'limegreen'])
        ax1.set_title('Топ-3 лучших месяцев', fontsize=14)
        ax1.set_ylabel('Процент проблемных рейсов (%)')
        ax1.set_xticks(range(len(best_months)))
        ax1.set_xticklabels([f"{row['Time.Month']}\n{row['Time.Month Name']}" 
                            for _, row in best_months.iterrows()])
        
        # Худшие месяцы  
        worst_months = monthly_data.tail(3)
        bars2 = ax2.bar(range(len(worst_months)), worst_months['Problem_Rate'],
                    color=['red', 'lightcoral', 'indianred'])
        ax2.set_title('Топ-3 худших месяцев', fontsize=14)
        ax2.set_ylabel('Процент проблемных рейсов (%)')
        ax2.set_xticks(range(len(worst_months)))
        ax2.set_xticklabels([f"{row['Time.Month']}\n{row['Time.Month Name']}" 
                            for _, row in worst_months.iterrows()])
        
        # Добавляем значения на столбцы
        for bars, ax in [(bars1, ax1), (bars2, ax2)]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        self._save_plot('1_best_worst_months.png')
        self._display_plot()

    def plot_airport_variance(self, variance_data: pd.DataFrame):
        """График для задания 2: Разброс по аэропортам"""
        if variance_data.empty:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Разброс проблемных рейсов по аэропортам', fontsize=16, fontweight='bold')
        
        # Аэропорты с наименьшим разбросом
        lowest_var = variance_data.head(3)
        bars1 = ax1.bar(lowest_var['Airport.Code'], lowest_var['Problem_Rate'],
                    color=['blue', 'lightblue', 'skyblue'])
        ax1.set_title('Топ-3 аэропортов с наименьшим разбросом\n(стабильные)', fontsize=14)
        ax1.set_ylabel('Стандартное отклонение (%)')
        
        # Аэропорты с наибольшим разбросом
        highest_var = variance_data.tail(3)
        bars2 = ax2.bar(highest_var['Airport.Code'], highest_var['Problem_Rate'],
                    color=['orange', 'gold', 'yellow'])
        ax2.set_title('Топ-3 аэропортов с наибольшим разбросом\n(нестабильные)', fontsize=14)
        ax2.set_ylabel('Стандартное отклонение (%)')
        
        # Добавляем значения
        for bars, ax in [(bars1, ax1), (bars2, ax2)]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        self._save_plot('2_airport_variance.png')
        self._display_plot()

    def plot_busiest_airports(self, busiest_data: Dict):
        """График для задания 3: Самый загруженный аэропорт"""
        if not busiest_data:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Топ-10 самых загруженных аэропортов
        top_airports = sorted(busiest_data['all_airports'].items(), 
                            key=lambda x: x[1], reverse=True)[:10]
        
        airports = [airport for airport, _ in top_airports]
        flights = [flights for _, flights in top_airports]
        
        bars = ax.bar(airports, flights, color=['red'] + ['lightblue']*9)
        
        ax.set_title('Топ-10 самых загруженных аэропортов', fontsize=16, fontweight='bold')
        ax.set_ylabel('Общее количество рейсов', fontsize=12)
        ax.set_xlabel('Код аэропорта', fontsize=12)
        
        # Подсветка самого загруженного
        bars[0].set_color('red')
        ax.text(0, flights[0] * 1.02, 'САМЫЙ ЗАГРУЖЕННЫЙ', 
            ha='center', va='bottom', fontweight='bold', color='red', fontsize=12)
        
        # Добавляем значения на столбцы
        for bar, flight_count in zip(bars, flights):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'{flight_count:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_plot('3_busiest_airports.png')
        self._display_plot()

    
    def plot_time_series_improved(self, time_series_data: pd.DataFrame):
        """График 3: Временные ряды и скользящее среднее (сохранение + показ)"""
        if time_series_data.empty:
            print("Нет данных для построения графика")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Убедимся, что данные отсортированы по дате
        time_series_data = time_series_data.sort_values('Date')
        
        # Исходные данные (тонкая линия)
        ax.plot(time_series_data['Date'], time_series_data['Delay_Rate'], 
                alpha=0.6, label='Процент задержек', linewidth=1, color='blue',
                marker='', linestyle='-')
        
        # Скользящее среднее - простая плавная линия
        moving_avg_column = None
        if 'Moving_Avg_3' in time_series_data.columns:
            moving_avg_column = 'Moving_Avg_3'
        elif 'Moving_Avg' in time_series_data.columns:
            moving_avg_column = 'Moving_Avg'
        
        if moving_avg_column:
            # Просто рисуем линию без интерполяции
            ax.plot(time_series_data['Date'], time_series_data[moving_avg_column],
                    linewidth=2, label='Скользящее среднее (3 месяца)', color='red',
                    alpha=0.9, marker='', linestyle='-')
        
        ax.set_title('Изменение процента задержек рейсов во времени\nсо скользящим средним', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Дата', fontsize=12)
        ax.set_ylabel('Процент задержек (%)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        self._save_plot('3_time_series_smooth.png')
        self._display_plot()

    def get_correlation_interpretation(self, correlation_value: float) -> str:
        """Интерпретация коэффициента корреляции"""
        abs_corr = abs(correlation_value)
        if abs_corr >= 0.8:
            return "очень сильная"
        elif abs_corr >= 0.6:
            return "сильная" 
        elif abs_corr >= 0.4:
            return "умеренная"
        elif abs_corr >= 0.2:
            return "слабая"
        else:
            return "очень слабая"