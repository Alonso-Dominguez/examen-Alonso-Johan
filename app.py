from flask import Flask, render_template, request, jsonify, send_file
import os
from controllers.data_controller import DataController
from utils.file_utils import save_uploaded_file, is_allowed_file
from io import StringIO, BytesIO
import numpy as np
import pandas as pd
from tkinter import filedialog, Tk
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

data_controller = DataController()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and is_allowed_file(file.filename):
        filepath = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
        success, result = data_controller.load_data(filepath)
        
        if success:
            return jsonify(result)
        else:
            return jsonify({'error': result}), 400
    else:
        return jsonify({'error': 'Invalid file type. Only CSV and JSON files are allowed.'}), 400

@app.route('/load_sample/<dataset_name>')
def load_sample(dataset_name):
    success, result = data_controller.load_sample_data(dataset_name)
    
    if success:
        return jsonify(result)
    else:
        return jsonify({'error': result}), 400

class DataController:
    def __init__(self):
        self.current_df = None
@app.route('/data/action', methods=['POST'])

def perform_data_action():
    action = request.json.get('action')
    params = request.json.get('params', {})
    
    success, result = data_controller.perform_action(action, params)
    
    if success:
        return jsonify(result)
    else:
        return jsonify({'error': result}), 400
    
def perform_action(self, action, params):
        try:
            if self.current_df is None:
                return False, 'No data loaded.'

            df = self.current_df.copy()

            if action == 'fillna':
                column = params.get('column')
                value = params.get('value')

                if column not in df.columns:
                    return False, f"Column '{column}' does not exist."

                # Reemplazar None con np.nan si es necesario
                df = df.replace({None: np.nan})

                # Convertir valores a numéricos si se desea
                if df[column].dtype == object:
                    df[column] = pd.to_numeric(df[column], errors='coerce')

                # Asegurarse de que el valor a llenar no sea None
                if value is None:
                    return False, 'No value provided for fillna.'

                # Intentar convertir el valor si es posible
                try:
                    value = float(value)
                except:
                    pass  # Dejarlo como string si no se puede convertir

                df[column] = df[column].fillna(value)

                self.current_df = df
                return True, {'message': f"Missing values in '{column}' filled with '{value}'."}

            return False, f"Unsupported action: {action}"

        except Exception as e:
            return False, f'Error performing action: {str(e)}'

@app.route('/visualization', methods=['POST'])
def generate_visualization():
    chart_type = request.json.get('chart_type')
    x_col = request.json.get('x_col')
    y_col = request.json.get('y_col', None)
    color_col = request.json.get('color_col', None)
    
    success, result = data_controller.generate_visualization(chart_type, x_col, y_col, color_col)
    
    if success:
        return jsonify(result)
    else:
        return jsonify({'error': result}), 400

@app.route('/download_csv')
def download_csv():
    if data_controller.current_df is None:
        return jsonify({'error': 'No data available to download'}), 400
    
    csv_buffer = StringIO()
    data_controller.current_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    return send_file(
        BytesIO(csv_buffer.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='cleaned_data.csv'
    )

@app.route('/download_json')
def download_json():
    if data_controller.current_df is None:
        return jsonify({'error': 'No data available to download'}), 400
    
    json_buffer = StringIO()
    data_controller.current_df.to_json(json_buffer, orient='records', indent=2)
    json_buffer.seek(0)
    
    return send_file(
        BytesIO(json_buffer.getvalue().encode('utf-8')),
        mimetype='application/json',
        as_attachment=True,
        download_name='cleaned_data.json'
    )

@app.route('/get_current_data')
def get_current_data():
    if data_controller.current_df is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    success, result = data_controller._prepare_data_response()
    return jsonify(result)

@app.route('/list_local_datasets')
def list_local_datasets():
    datasets = data_controller.list_local_datasets(app.config['UPLOAD_FOLDER'])
    return jsonify({'datasets': datasets})

@app.route('/load_local_dataset/<filename>')
def load_local_dataset(filename):
    success, result = data_controller.load_local_dataset(filename, app.config['UPLOAD_FOLDER'])
    if success:
        return jsonify(result)
    else:
        return jsonify({'error': result}), 400

if __name__ == '__main__':
    app.run(debug=True)
