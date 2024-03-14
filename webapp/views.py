from django.shortcuts import render
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from django.http import HttpResponse
import os
from .models import Actions





        
#         new_model = load_model("webapp/models/newMobilenet.h5", compile=False)
#         test_image = image.load_img(path, target_size=(256, 256))
#         test_image = image.img_to_array(test_image)
#         test_image /= 255
#         a = acc.iloc[- 1, 1]

#         # else:
#         #     new_model = load_model("webapp/models/MobileNet.h5", compile=False)
#         #     test_image = image.load_img(path, target_size=(224, 224))
#         #     test_image = image.img_to_array(test_image)
#         #     test_image /= 255
#         #     a = acc.iloc[m-1, 1]

#         test_image = np.expand_dims(test_image, axis=0)
#         result = new_model.predict(test_image)
#         pred = Action[np.argmax(result)]
#         print(pred)

#         return render(request, resultpage, {'text': pred, 'path': 'static/images/'+fn.filename(), 'a': round(a*100, 3)})

#     return render(request, resultpage)


# def prediction(request):
#     livestreaming()
#     print("Running")
#     return render(request, homepage)
from django.shortcuts import render
import pandas as pd
import numpy as np
from django.conf import settings
from PIL import Image
import yolov5

# from .prediction import *



# Create your views here.

homepage = 'index.html'
resultpage = 'result.html'


Action = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']


def index(request):
    return render(request, homepage)


def result(request):
    if request.method == 'POST':
        # m = int(request.POST['alg'])
        file = request.FILES['file']
        fn = Actions(images=file)
        fn.save()
        path = os.path.join('webapp/static/images/',fn.filename())
        acc = pd.read_csv("webapp/Accurary.csv")

        
        new_model = load_model("webapp/models/newMobilenet.h5", compile=False)
        test_image = image.load_img(path, target_size=(256, 256))
        test_image = image.img_to_array(test_image)
        test_image /= 255
        a = acc.iloc[- 1, 1]

       
        model = yolov5.load('keremberke/yolov5m-garbage')

        model.conf = 0.25
        model.iou = 0.45 
        model.agnostic = False 
        model.multi_label = False
        model.max_det = 1000 

        results = model(path, size=640)

        results = model(path, augment=True)
        
        numpy_image = results.render()[0]
        output_image = Image.fromarray(numpy_image)

       
        path = os.path.join('webapp/static/results/',fn.filename())
        output_image.save(path)

        return render(request, resultpage, {'text': results.pandas().xyxy[0]["name"].to_json(orient='records'), 'path':'static/results/'+fn.filename() })

    return render(request, resultpage)


