from flask import Flask, request, render_template
import cv2
import numpy as np
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transform', methods=['POST'])
def transform():
    # Obtener la imagen cargada desde el formulario
    image_data = request.files['image'].read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Aplicar transformaciones según los valores recibidos del formulario
    rotation = int(request.form['rotation'])
    horizontal_flip = int(request.form['horizontal_flip'])
    vertical_flip = int(request.form['vertical_flip'])
    grayscalex = int(request.form['grayscalex'])
    grayscaley = int(request.form['grayscaley'])
    escalagris = int(request.form['escalagris'])
   
    if rotation != 0:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if rotation >=91:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            if rotation >=181:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                if rotation >=241:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        

    if horizontal_flip != 0:
        matriz_traslacion = np.float32([[1, 0, horizontal_flip], [0, 1, horizontal_flip]])
        image = cv2.warpAffine(image, matriz_traslacion, (horizontal_flip, horizontal_flip))

    if vertical_flip != 0 :
        image = cv2.resize(image, None, fx=vertical_flip, fy=vertical_flip)

    if grayscalex !=0:
        matriz_cizallado = np.float32([[1, grayscalex, 0], [grayscaley, 1, 0]])
        image = cv2.warpAffine(image, matriz_cizallado, (grayscalex, grayscaley))
    
    if grayscaley !=0:
        matriz_cizallado = np.float32([[1, grayscalex, 0], [grayscaley, 1, 0]])
        image = cv2.warpAffine(image, matriz_cizallado, (grayscalex, grayscaley))
        
    if escalagris !=0:
        imagen_transparente = np.zeros(image.shape, dtype=np.uint8)
        image = cv2.addWeighted(image, escalagris, imagen_transparente, 1 - escalagris, 0, imagen_transparente)
    
    # Codificar la imagen resultante en base64 para mostrarla en la vista previa
    _, buffer = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return encoded_image

if __name__ == '__main__':
    app.run(debug=True)