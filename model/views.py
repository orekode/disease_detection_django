import os
import numpy as np
from PIL import Image
import torch
from . import CNN
import numpy as np
import pandas as pd

import torchvision.transforms.functional as TF
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


def gen_path(path):
    return os.path.join(current_directory, path)



current_directory = os.path.dirname(os.path.abspath(__file__))

disease_info = pd.read_csv(gen_path('disease_info.csv'), encoding='cp1252')
supplement_info = pd.read_csv(gen_path('supplement_info.csv'), encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load(gen_path("plant_disease_model_1_latest.pt")))
model.eval()

# def prediction(image_path):
#     image = Image.open(image_path)
#     image = image.resize((224, 224))
#     input_data = TF.to_tensor(image)
#     input_data = input_data.view((-1, 3, 224, 224))
#     output = model(input_data)
#     output = output.detach().numpy()
#     index = np.argmax(output)
#     return index

def prediction(image_path):
    image = Image.open(image_path).convert('RGB')  # Ensure the image has 3 channels (RGB)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.unsqueeze(0)  # Add batch dimension
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

@csrf_exempt
def submit(request):
    if request.method == 'POST':
        image = request.FILES['image']
        file_path = default_storage.save(os.path.join('uploads', image.name), image)
        full_file_path = os.path.join(settings.MEDIA_ROOT, file_path)

        pred = prediction(full_file_path)
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        response_data = {
            'title': str(title),
            'description': str(description),
            'prevent': str(prevent),
            'image_url': str(image_url),
            'pred': int(pred),  # Convert to native int
            'supplement_name': str(supplement_name),
            'supplement_image_url': str(supplement_image_url),
            'buy_link': str(supplement_buy_link),
        }

        print(response_data)

        return JsonResponse(response_data)
    return JsonResponse({'error': 'Invalid request method'}, status=400)
