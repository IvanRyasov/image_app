import os
import io
import base64
import random
from flask import Flask, render_template, request, session, redirect, url_for, flash
from PIL import Image
import matplotlib
matplotlib.use('Agg') # Для работы matplotlib на сервере без GUI
import matplotlib.pyplot as plt

# Для нейросети
import torch
from torchvision import models, transforms
import json
import urllib.request

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")

# Загрузка легковесной нейросети для классификации (MobileNet V2)
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Загрузка классов ImageNet
url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
try:
    with urllib.request.urlopen(url) as response:
        imagenet_classes = json.loads(response.read().decode())
except:
    imagenet_classes = ["Неизвестный класс"] * 1000

def get_image_base64(img):
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG', quality=90)
    img_io.seek(0)
    return base64.b64encode(img_io.getvalue()).decode('ascii')

def generate_histogram(img):
    plt.figure(figsize=(4, 3))
    colors = ('r', 'g', 'b')
    for i, col in enumerate(colors):
        # Получаем данные канала и строим график
        hist = img.histogram()[i*256:(i+1)*256]
        plt.plot(hist, color=col)
    
    plt.title("Распределение цветов")
    plt.xlabel("Интенсивность")
    plt.ylabel("Пиксели")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')

def classify_image(img):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img.convert('RGB'))
    input_batch = input_tensor.unsqueeze(0) 

    with torch.no_grad():
        output = model(input_batch)
    
    _, index = torch.max(output, 1)
    return imagenet_classes[index[0].item()]

@app.route('/', methods=['GET'])
def index():
    # Простейшая математическая капча
    num1, num2 = random.randint(1, 10), random.randint(1, 10)
    session['captcha_answer'] = str(num1 + num2)
    return render_template('index.html', num1=num1, num2=num2)

@app.route('/process', methods=['POST'])
def process():
    captcha_input = request.form.get('captcha')
    if captcha_input != session.get('captcha_answer'):
        flash("Неверная капча! Попробуйте снова.")
        return redirect(url_for('index'))

    if 'image' not in request.files:
        flash("Файл не загружен")
        return redirect(url_for('index'))

    file = request.files['image']
    scale = float(request.form.get('scale', 1.0))

    if file.filename == '':
        return redirect(url_for('index'))

    # Чтение оригинального изображения
    orig_img = Image.open(file.stream).convert('RGB')
    orig_hist = generate_histogram(orig_img)
    orig_b64 = get_image_base64(orig_img)

    # Классификация (Нейросеть)
    ai_class = classify_image(orig_img)

    # Изменение размера
    new_size = (int(orig_img.width * scale), int(orig_img.height * scale))
    resized_img = orig_img.resize(new_size)
    
    resized_hist = generate_histogram(resized_img)
    resized_b64 = get_image_base64(resized_img)

    return render_template('result.html', 
                           orig_img=orig_b64, orig_hist=orig_hist, 
                           res_img=resized_b64, res_hist=resized_hist,
                           scale=scale, ai_class=ai_class,
                           orig_size=orig_img.size, new_size=new_size)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')