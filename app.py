import os
import logging
from flask import Flask, render_template, request, redirect, url_for
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from io import BytesIO
import base64

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx'}

# Define datasets
datasets = {
    'Dataset1': pd.DataFrame({
        'Mes': ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio'],
        'Ventas_A': [150, 200, 250, 300, 350, 400],
        'Ventas_B': [100, 150, 200, 250, 300, 350]
    }),
    'Dataset2': pd.DataFrame({
        'Ciudad': ['Ciudad1', 'Ciudad2', 'Ciudad3', 'Ciudad4'],
        'Temp_Max': [30, 35, 32, 31],
        'Temp_Min': [20, 25, 22, 21]
    }),
    'Dataset3': pd.DataFrame({
        'Año': [2010, 2011, 2012, 2013, 2014, 2015],
        'Población_Urbana': [50000, 52000, 54000, 56000, 58000, 60000],
        'Población_Rural': [20000, 19000, 18000, 17000, 16000, 15000]
    })
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    logging.debug("Renderizando la página de inicio.")
    return render_template('index.html', datasets=datasets.keys())

@app.route('/upload', methods=['POST'])
def upload_file():
    logging.debug("Intentando cargar un archivo.")
    if 'file' not in request.files:
        logging.debug("No se encontró el archivo en la solicitud.")
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        logging.debug("No se seleccionó ningún archivo.")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Ensure the 'uploads' directory exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        file.save(file_path)
        logging.debug(f"Archivo {filename} cargado con éxito.")
        return redirect(url_for('auto_generate', filename=filename))
    return redirect(request.url)

@app.route('/auto_generate', methods=['GET'])
def auto_generate_dataset():
    dataset_name = request.args.get('dataset')
    if dataset_name not in datasets:
        return "Error: Dataset no encontrado."
    df = datasets[dataset_name]
    plot_urls = auto_generate_plots(df)
    return render_template('plot.html', plot_urls=plot_urls)

@app.route('/auto_generate/<filename>', methods=['GET'])
def auto_generate(filename):
    logging.debug(f"Generando gráficos automáticamente para el archivo: {filename}")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if filename.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    plot_urls = auto_generate_plots(df)
    return render_template('plot.html', plot_urls=plot_urls)

def auto_generate_plots(df):
    plot_urls = []
    num_columns = df.select_dtypes(include=np.number).columns.tolist()
    cat_columns = df.select_dtypes(exclude=np.number).columns.tolist()

    if len(num_columns) >= 2:
        x_column = num_columns[0]
        y_column = num_columns[1]
        plot_types = ['line', 'bar', 'scatter', 'histogram', 'area', '3d', 'animated']
    elif len(num_columns) == 1 and len(cat_columns) >= 1:
        x_column = cat_columns[0]
        y_column = num_columns[0]
        plot_types = ['bar']
    else:
        return []

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    markers = ['o', 's', 'D', '^', 'v', '<', '>']
    linestyles = ['-', '--', '-.', ':']

    for plot_type in plot_types:
        img = BytesIO()
        try:
            logging.debug(f"Generando gráfico tipo: {plot_type}")
            color = colors[plot_types.index(plot_type) % len(colors)]
            marker = markers[plot_types.index(plot_type) % len(markers)]
            linestyle = linestyles[plot_types.index(plot_type) % len(linestyles)]

            if plot_type == 'line':
                plt.figure(figsize=(10, 6))
                plt.plot(df[x_column], df[y_column], marker=marker, color=color, linestyle=linestyle, label=f'{y_column} vs {x_column}')
                plt.title(f'{y_column} vs {x_column}')
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.grid(True)
                plt.legend()
            
            elif plot_type == 'bar':
                plt.figure(figsize=(10, 6))
                bar_width = 0.35
                index = np.arange(len(df[x_column]))
                plt.bar(index, df[y_column], bar_width, color=color, label=f'{y_column}')
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.title(f'{y_column} vs {x_column}')
                plt.xticks(index, df[x_column], rotation=45)
                plt.legend()
            
            elif plot_type == 'scatter':
                plt.figure(figsize=(10, 6))
                plt.scatter(df[x_column], df[y_column], color=color, marker=marker, label=f'{y_column} vs {x_column}')
                plt.title(f'{y_column} vs {x_column}')
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.grid(True)
                plt.legend()
            
            elif plot_type == 'histogram':
                plt.figure(figsize=(10, 6))
                plt.hist(df[y_column], bins=10, alpha=0.5, color=color, label=f'{y_column}')
                plt.title(f'Distribución de {y_column}')
                plt.xlabel(y_column)
                plt.ylabel('Frecuencia')
                plt.legend()
            
            elif plot_type == 'area':
                plt.figure(figsize=(10, 6))
                plt.fill_between(df[x_column], df[y_column], color=color, alpha=0.5, label=f'{y_column}')
                plt.title(f'{y_column} vs {x_column}')
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.legend()

            elif plot_type == '3d':
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(df[x_column], df[y_column], zs=0, zdir='z', marker=marker, color=color)
                ax.set_title(f'{y_column} vs {x_column}')
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                ax.set_zlabel('Z')
            
            elif plot_type == 'animated':
                fig, ax = plt.subplots(figsize=(10, 6))
                def animate(i):
                    ax.clear()
                    ax.plot(df[x_column][:i+1], df[y_column][:i+1], marker=marker, color=color, linestyle=linestyle)
                    ax.set_title(f'{y_column} vs {x_column}')
                    ax.set_xlabel(x_column)
                    ax.set_ylabel(y_column)
                    ax.grid(True)
                    ax.legend([f'{y_column} vs {x_column}'])
                ani = animation.FuncAnimation(fig, animate, frames=len(df), repeat=False)

            else:
                continue

            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
            plot_urls.append(plot_url)
            logging.debug(f"Gráfico tipo: {plot_type} generado con éxito.")

        except KeyError as e:
            logging.error(f"Error al generar gráfico: {e}")
            continue

    return plot_urls

if __name__ == '__main__':
    app.run(debug=True)
