import os
from werkzeug.utils import secure_filename

# Guarda un archivo subido en una carpeta específica
def save_uploaded_file(file, upload_folder):
    # Crea la carpeta si no existe
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # Asegura un nombre de archivo seguro
    filename = secure_filename(file.filename)

    # Ruta completa del archivo a guardar
    filepath = os.path.join(upload_folder, filename)

    # Guarda el archivo
    file.save(filepath)
    
    return filepath

# Verifica si el archivo tiene una extensión permitida
def is_allowed_file(filename):
    allowed_extensions = {'csv', 'json'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
