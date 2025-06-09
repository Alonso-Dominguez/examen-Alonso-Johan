import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
import base64
import json
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

class DataController:
    def __init__(self):
        self.current_data = None
        self.current_df = None
        self.sample_datasets = {
            'iris': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv',
            'tips': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv',
            'titanic': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv',
            'mpg': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv'
        }

    def load_data(self, filepath):
        try:
            if filepath.endswith('.csv'):
                self.current_df = pd.read_csv(filepath)
            elif filepath.endswith('.json'):
                self.current_df = pd.read_json(filepath)
            else:
                return False, "Formato de archivo no soportado. Use CSV o JSON."
                
            self.current_data = self._prepare_data_response()
            return True, self.current_data
        except Exception as e:
            return False, str(e)

    def load_sample_data(self, dataset_name):
        try:
            if dataset_name in self.sample_datasets:
                url = self.sample_datasets[dataset_name]
                self.current_df = pd.read_csv(url)
                self.current_data = self._prepare_data_response()
                return True, self.current_data
            else:
                return False, f"Dataset '{dataset_name}' not available"
        except Exception as e:
            return False, str(e)

    def perform_action(self, action, params):
        if self.current_df is None:
            return False, "No data loaded"
        try:
            result_df = self.current_df.copy()
            # Normalizar parámetros para columnas múltiples
            if 'columns' in params and isinstance(params['columns'], str):
                # Si viene como string de un solo valor
                params['columns'] = [params['columns']]
            if action == 'show-head':
                n = int(params.get('rows', 10))
                result_df = self.current_df.head(n)
            elif action == 'show-tail':
                n = int(params.get('rows', 10))
                result_df = self.current_df.tail(n)
            elif action == 'show-info':
                info_data = {
                    'summary': {
                        'Filas': len(self.current_df),
                        'Columnas': len(self.current_df.columns),
                        'Memoria': f"{self.current_df.memory_usage(deep=True).sum() / 1024:.2f} KB"
                    },
                    'columns': []
                }
                for col in self.current_df.columns:
                    col_info = {
                        'name': col,
                        'type': str(self.current_df[col].dtype),
                        'non_null': int(self.current_df[col].notna().sum()),
                        'null_percent': f"{self.current_df[col].isna().mean() * 100:.1f}%",
                        'unique': int(self.current_df[col].nunique())
                    }
                    info_data['columns'].append(col_info)
                return True, {'info': info_data}
            elif action == 'show-describe':
                description = self.current_df.describe().to_dict()
                return True, {'description': description}
            elif action == 'select-column':
                column = params.get('column')
                if column in self.current_df.columns:
                    result_df = self.current_df[[column]]
                else:
                    return False, f"Column '{column}' not found"
            elif action == 'filter-data':
                column = params.get('column')
                operator = params.get('operator')
                value = params.get('value')
                try:
                    if operator == '==':
                        result_df = self.current_df[self.current_df[column] == value]
                    elif operator == '!=':
                        result_df = self.current_df[self.current_df[column] != value]
                    elif operator == '>':
                        result_df = self.current_df[self.current_df[column].astype(float) > float(value)]
                    elif operator == '<':
                        result_df = self.current_df[self.current_df[column].astype(float) < float(value)]
                    elif operator == '>=':
                        result_df = self.current_df[self.current_df[column].astype(float) >= float(value)]
                    elif operator == '<=':
                        result_df = self.current_df[self.current_df[column].astype(float) <= float(value)]
                    elif operator == 'contains':
                        result_df = self.current_df[self.current_df[column].astype(str).str.contains(value, na=False)]
                    elif operator == 'startsWith':
                        result_df = self.current_df[self.current_df[column].astype(str).str.startswith(value, na=False)]
                    elif operator == 'endsWith':
                        result_df = self.current_df[self.current_df[column].astype(str).str.endswith(value, na=False)]
                    else:
                        return False, f"Unsupported operator: {operator}"
                except Exception as e:
                    return False, f"Error applying filter: {str(e)}"
            elif action == 'fill-na':
                column = params.get('column')
                fill_value = params.get('fill-value') or params.get('fill_value')
                try:
                    dtype = self.current_df[column].dtype
                    if np.issubdtype(dtype, np.number):
                        fill_value = float(fill_value)
                    elif np.issubdtype(dtype, np.bool_):
                        fill_value = bool(fill_value)
                    self.current_df[column] = self.current_df[column].fillna(fill_value)
                    result_df = self.current_df
                except Exception as e:
                    return False, f"Error filling NA values: {str(e)}"
            elif action == 'drop-na':
                columns = params.get('columns', [])
                if isinstance(columns, str):
                    columns = [columns]
                threshold = float(params.get('threshold', 100)) / 100
                try:
                    if columns:
                        result_df = self.current_df.dropna(subset=columns, thresh=int(len(columns)*threshold))
                    else:
                        result_df = self.current_df.dropna(thresh=int(len(self.current_df.columns)*threshold))
                except Exception as e:
                    return False, f"Error dropping NA values: {str(e)}"
            elif action == 'drop-duplicates':
                columns = params.get('columns', [])
                if isinstance(columns, str):
                    columns = [columns]
                keep_first = params.get('keep-first', True)
                try:
                    subset = columns if columns else None
                    keep = 'first' if keep_first else False
                    result_df = self.current_df.drop_duplicates(subset=subset, keep=keep)
                except Exception as e:
                    return False, f"Error dropping duplicates: {str(e)}"
            elif action == 'convert-types':
                column = params.get('column')
                new_type = params.get('type')
                try:
                    if new_type == 'number':
                        self.current_df[column] = pd.to_numeric(self.current_df[column], errors='coerce')
                    elif new_type == 'string':
                        self.current_df[column] = self.current_df[column].astype(str)
                    elif new_type == 'boolean':
                        self.current_df[column] = self.current_df[column].astype(bool)
                    elif new_type == 'date':
                        self.current_df[column] = pd.to_datetime(self.current_df[column], errors='coerce')
                    else:
                        return False, f"Unsupported type: {new_type}"
                    result_df = self.current_df
                except Exception as e:
                    return False, f"Error converting type: {str(e)}"
            elif action == 'calculate-stats':
                column = params.get('column')
                if column not in self.current_df.columns:
                    return False, f"Column '{column}' not found"
                if not np.issubdtype(self.current_df[column].dtype, np.number):
                    return False, f"Column '{column}' is not numeric"
                stats = {
                    'mean': float(self.current_df[column].mean()),
                    'median': float(self.current_df[column].median()),
                    'std': float(self.current_df[column].std()),
                    'min': float(self.current_df[column].min()),
                    'max': float(self.current_df[column].max()),
                    'count': int(self.current_df[column].count()),
                    'q1': float(self.current_df[column].quantile(0.25)),
                    'q3': float(self.current_df[column].quantile(0.75))
                }
                return True, {'stats': stats}
            else:
                return False, f"Unknown action: {action}"
            if action in ['fill-na', 'drop-na', 'drop-duplicates', 'convert-types']:
                self.current_df = result_df.copy()
                self.current_data = self._prepare_data_response()
            return True, self._prepare_data_response(result_df)
        except Exception as e:
            return False, str(e)

    def generate_visualization(self, chart_type, x_col, y_col=None, color_col=None):
        if self.current_df is None:
            return False, "No data loaded"
        
        try:
            plt.figure()
            
            if chart_type == 'histogram':
                self.current_df[x_col].hist()
                plt.title(f'Histogram of {x_col}')
                plt.xlabel(x_col)
                plt.ylabel('Frequency')
            elif chart_type == 'bar':
                if color_col:
                    grouped = self.current_df.groupby(color_col)[x_col].mean()
                    grouped.plot(kind='bar')
                    plt.title(f'Average {x_col} by {color_col}')
                else:
                    self.current_df[x_col].value_counts().plot(kind='bar')
                    plt.title(f'Count of {x_col}')
                plt.xlabel(x_col)
            elif chart_type == 'pie':
                self.current_df[x_col].value_counts().plot(kind='pie', autopct='%1.1f%%')
                plt.title(f'Distribution of {x_col}')
                plt.ylabel('')
            elif chart_type == 'line':
                if y_col:
                    self.current_df.plot(x=x_col, y=y_col, kind='line')
                    plt.title(f'{y_col} vs {x_col}')
                else:
                    return False, "Line chart requires y-axis column"
            elif chart_type == 'scatter':
                if y_col:
                    self.current_df.plot(x=x_col, y=y_col, kind='scatter')
                    plt.title(f'{y_col} vs {x_col}')
                else:
                    return False, "Scatter plot requires y-axis column"
            elif chart_type == 'box':
                self.current_df.boxplot(column=x_col)
                plt.title(f'Boxplot of {x_col}')
            else:
                return False, f"Unsupported chart type: {chart_type}"
            
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight')
            plt.close()
            img_buffer.seek(0)
            
            img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            return True, {'image': img_str}
        except Exception as e:
            return False, str(e)

    def list_local_datasets(self, uploads_folder='uploads'):
        import os
        datasets = []
        for fname in os.listdir(uploads_folder):
            if fname.lower().endswith('.csv'):
                datasets.append(fname)
        return datasets

    def load_local_dataset(self, filename, uploads_folder='uploads'):
        import os
        filepath = os.path.join(uploads_folder, filename)
        if not os.path.exists(filepath):
            return False, f"El archivo {filename} no existe."
        try:
            # Intentar diferentes codificaciones y delimitadores
            try:
                df = pd.read_csv(filepath)
            except Exception:
                try:
                    df = pd.read_csv(filepath, encoding='latin1')
                except Exception:
                    df = pd.read_csv(filepath, delimiter=';', encoding='utf-8', engine='python')
            self.current_df = df
            self.current_data = self._prepare_data_response()
            return True, self.current_data
        except Exception as e:
            return False, f"Error al leer el archivo: {str(e)}"

    def _prepare_data_response(self, df=None):
        if df is None:
            df = self.current_df
            
        data = []
        for row in df.values:
            processed_row = []
            for val in row:
                if pd.isna(val):
                    processed_row.append('NaN')
                elif isinstance(val, (np.int64, np.float64)):
                    processed_row.append(float(val))
                else:
                    processed_row.append(str(val))
            data.append(processed_row)
        
        return {
            'columns': df.columns.tolist(),
            'data': data,
            'shape': {'rows': df.shape[0], 'cols': df.shape[1]},
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }