from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import subprocess
import yaml

app = Flask(__name__)

# Define base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

# Define paths
INPUT_FOLDER = os.path.join(PARENT_DIR, 'data/input')
OUTPUT_FOLDER = os.path.join(PARENT_DIR, 'data/output')
LOG_FOLDER = os.path.join(PARENT_DIR, 'logs')
LOG_FILE = os.path.join(LOG_FOLDER, 'app.log')

app.config['INPUT_FOLDER'] = INPUT_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['LOG_FOLDER'] = LOG_FOLDER
app.config["PARENT_DIR"] = PARENT_DIR

# Ensure directories exist
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# Route to render the form
@app.route('/')
def index():
    return render_template('index.html', download_link=None)

# Route to handle form submission
@app.route('/run_script', methods=['POST'])
def run_script():
    config_path = os.path.join(app.config['INPUT_FOLDER'], 'config.yaml')

    if 'config_file' in request.files:
        config_file = request.files['config_file']
        config_file.save(config_path)

    if 'data_file' in request.files:
        data_file = request.files['data_file']
        data_path = os.path.join(app.config['INPUT_FOLDER'], 'source_data.xlsx')
        data_file.save(data_path)

    # Load the config file to get the output file path
    try:
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
            output_file_path = config['data_paths']['output_file']
            output_file_abs_path = os.path.join(app.config["PARENT_DIR"], output_file_path)

            # Set the working directory to the parent directory
            cwd = app.config["PARENT_DIR"]

            # Run the main script
            subprocess.run(['python', os.path.join('scripts', 'main.py'), '--config', config_path], check=True, cwd=cwd)

            # Check if the output file exists
            if os.path.exists(output_file_abs_path):
                download_link = url_for('download_file', filename=os.path.basename(output_file_abs_path))
                return render_template('index.html', download_link=download_link)
            else:
                return 'Output file not found'

    except subprocess.CalledProcessError as e:
        return f'An error occurred while running the script: {e}'
    except Exception as e:
        return f'An error occurred while loading the config file: {e}'

@app.route('/data/output/<filename>')
def download_file(filename):
    with open(os.path.join(app.config['INPUT_FOLDER'], 'config.yaml'), 'r') as stream:
        config = yaml.safe_load(stream)
        output_file_path = config['data_paths']['output_file']
    directory = os.path.dirname(os.path.join(app.config["PARENT_DIR"], output_file_path))
    return send_from_directory(directory, filename)

# Route to display the config file
@app.route('/view_config')
def view_config():
    config_path = os.path.join(app.config['INPUT_FOLDER'], 'config.yaml')
    try:
        with open(config_path, 'r') as stream:
            config_content = yaml.safe_load(stream)
            return render_template('config.html', config_content=config_content)
    except Exception as e:
        return f'Error reading config file: {e}'

@app.route('/file_status')
def file_status():
    config_path = os.path.join(app.config['INPUT_FOLDER'], 'config.yaml')
    try:
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
            output_file_path = config['data_paths']['output_file']
            output_file_path = os.path.join(app.config['OUTPUT_FOLDER'], output_file_path)

        file_exists = os.path.exists(output_file_path)
        return jsonify({'file_exists': file_exists})
    except Exception as e:
        return jsonify({'file_exists': False})

# Route to fetch and display logs
@app.route('/logs')
def logs():
    try:
        with open(LOG_FILE, 'r') as log_file:
            log_content = log_file.read()
        return log_content
    except Exception as e:
        return f'Error reading log file: {e}'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port, debug=True)
