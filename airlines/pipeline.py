import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from typing import Generator, Dict, Any, Union, List

class DataProcessingPipeline:
    def __init__(self, chunksize: int = 400):
        self.chunksize = chunksize
    
    def read_csv_chunks(self, file_path: str) -> Generator[pd.DataFrame, None, None]:
        """Генератор для чтения CSV файла по частям"""
        print(f"Чтение CSV файла: {file_path}")
        try:
            for chunk in pd.read_csv(file_path, chunksize=self.chunksize):
                yield chunk
        except Exception as e:
            print(f"Ошибка чтения CSV: {e}")
    
    def filter_columns(self, chunks: Generator[pd.DataFrame, None, None], 
                      columns: list) -> Generator[pd.DataFrame, None, None]:
        """Генератор для фильтрации столбцов"""
        for chunk in chunks:
            available_columns = [col for col in columns if col in chunk.columns]
            yield chunk[available_columns]
    
    def clean_data(self, chunks: Generator[pd.DataFrame, None, None]) -> Generator[pd.DataFrame, None, None]:
        """Генератор для очистки данных"""
        for chunk in chunks:
            # Удаляем строки с пропущенными значениями в ключевых столбцах
            key_columns = ['Time.Year', 'Time.Month', 'Statistics.Flights.Total']
            available_key_cols = [col for col in key_columns if col in chunk.columns]
            if available_key_cols:
                chunk_clean = chunk.dropna(subset=available_key_cols)
                yield chunk_clean
            else:
                yield chunk
    
    def aggregate_chunks(self, chunks: Generator[pd.DataFrame, None, None], 
                    group_by: Union[str, List[str]], agg_columns: Dict[str, Any]) -> pd.DataFrame:
        """Простая и эффективная агрегация данных"""
        
        # Собираем все данные в один DataFrame
        #all_data = []
        
        for chunk in chunks:
            # Проверяем наличие нужных столбцов
            required_columns = [group_by] if isinstance(group_by, str) else group_by
            required_columns.extend(agg_columns.keys())
            
            if all(col in chunk.columns for col in required_columns):
                #all_data.append(chunk[required_columns])
                filtered_chunk = chunk[required_columns]
                combined_data = pd.concat([combined_data, filtered_chunk], ignore_index=True)

        
        # Объединяем и агрегируем
        #combined_data = pd.concat(all_data, ignore_index=True)
        result = combined_data.groupby(group_by).agg(agg_columns).reset_index()
        
        return result
    
    def aggregate_chunks_incremental(self, chunks: Generator[pd.DataFrame, None, None], 
                                   group_by: Union[str, List[str]], agg_columns: Dict[str, Any]) -> pd.DataFrame:
        """
        Альтернативная версия: инкрементальная агрегация
        Более эффективна для большого количества чанков
        """
        from collections import defaultdict
        import numpy as np
        
        # Создаем ключ для группировки
        def get_group_key(row, group_cols):
            if isinstance(group_cols, list):
                return tuple(row[col] for col in group_cols)
            else:
                return row[group_cols]
        
        # Словарь для хранения промежуточных результатов
        result_dict = defaultdict(lambda: {col: 0 for col in agg_columns.keys()})
        
        for chunk in chunks:
            # Проверяем наличие всех группирующих столбцов
            if isinstance(group_by, list):
                group_columns_present = all(col in chunk.columns for col in group_by)
            else:
                group_columns_present = group_by in chunk.columns
            
            if group_columns_present:
                # Агрегируем текущий чанк
                aggregated_chunk = chunk.groupby(group_by).agg(agg_columns).reset_index()
                
                # Обновляем общий результат
                for _, row in aggregated_chunk.iterrows():
                    group_key = get_group_key(row, group_by)
                    for col in agg_columns.keys():
                        if col in row:
                            result_dict[group_key][col] += row[col]
        
        # Конвертируем словарь в DataFrame
        if result_dict:
            result_data = []
            for group_key, aggregates in result_dict.items():
                if isinstance(group_by, list):
                    row_data = {col: val for col, val in zip(group_by, group_key)}
                else:
                    row_data = {group_by: group_key}
                row_data.update(aggregates)
                result_data.append(row_data)
            
            return pd.DataFrame(result_data)
        
        return pd.DataFrame()
    
    def csv_to_parquet(self, csv_file: str, parquet_file: str):
        """Конвертация CSV в Parquet с использованием потокового чтения"""
        try:
            # Читаем CSV чанками и записываем в Parquet
            first_chunk = True
            schema = None
            
            for chunk in self.read_csv_chunks(csv_file):
                if first_chunk:
                    # Определяем схему по первому чанку
                    table = pa.Table.from_pandas(chunk)
                    schema = table.schema
                    with pq.ParquetWriter(parquet_file, schema) as writer:
                        writer.write_table(table)
                    first_chunk = False
                else:
                    # Добавляем последующие чанки
                    table = pa.Table.from_pandas(chunk, schema=schema)
                    with pq.ParquetWriter(parquet_file, schema) as writer:
                        writer.write_table(table)
            
            print(f"Создан Parquet файл: {parquet_file}")
        except Exception as e:
            print(f"Ошибка при конвертации в Parquet: {e}")