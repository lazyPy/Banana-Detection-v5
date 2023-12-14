from django.shortcuts import render
from .forms import ImageUploadForm
import cv2
import numpy as np
import base64
from collections import defaultdict
# from django.conf import settings
from django.apps import apps

# Accessing initialized elements from the app configuration
interpreter = apps.get_app_config('app').interpreter
input_details = apps.get_app_config('app').input_details
output_details = apps.get_app_config('app').output_details
names = apps.get_app_config('app').names
colors = apps.get_app_config('app').colors


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


def perform_banana_detection(img):
    # bucket_name = settings.AWS_STORAGE_BUCKET_NAME
    # file_name = 'best.tflite'
    #
    # # Initialize S3 client
    # s3_client = settings.S3_CLIENT
    #
    # # Load the TFLite model from S3
    # response = s3_client.get_object(Bucket=bucket_name, Key=file_name)
    # tflite_model_content = response['Body'].read()
    # interpreter = tf.lite.Interpreter(model_content=tflite_model_content)

    # Perform letterboxing and preprocessing
    image = img.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255

    # Test the model on random input data
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], im)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data
    output_data = interpreter.get_tensor(output_details[0]['index'])

    ori_images = [img.copy()]

    # Initialize a count variable to keep track of detected objects
    count_objects = 0
    # Initialize a dictionary to count the occurrences of each class
    count_classes = defaultdict(int)

    for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(output_data):
        image = ori_images[int(batch_id)]
        box = np.array([x0, y0, x1, y1])
        box -= np.array(dwdh * 2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score), 3)
        name = names[cls_id]
        color = colors[name]
        name += ' ' + str(score)

        # Draw and count object and class if score is higher than 0.6
        if score > 0.6:
            cv2.rectangle(image, box[:2], box[2:], color, 2)
            cv2.putText(image, name, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)
            count_objects += 1
            count_classes[names[cls_id]] += 1

    # Ensure count_classes is not empty before finding the majority class
    if count_classes:
        majority_class = max(count_classes, key=count_classes.get)
    else:
        majority_class = "No banana detected"

    # Return the processed image, count of detected objects, and the majority detected class
    return ori_images[0], count_objects, majority_class


def detect_banana(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = form.cleaned_data['image']

            # Perform detection on the uploaded image

            # Read the image as a NumPy array
            img = np.fromstring(image_file.read(), np.uint8)

            # Decode the image with OpenCV
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)

            # Define the maximum width and height
            max_width = 640
            max_height = 640

            # Use Pillow for efficient image resizing
            from PIL import Image
            img = Image.fromarray(img)
            new_size = max_width, max_height

            # Resize the image while maintaining aspect ratio
            img.thumbnail(new_size, Image.ANTIALIAS)

            # Convert the resized image back to NumPy array
            img = np.array(img)

            # Convert color space from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Perform banana detection and get processed image and count
            processed_image, count_objects, majority_class = perform_banana_detection(img)

            # Convert the processed image to Base64
            _, buffer = cv2.imencode('.png', cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
            prediction = base64.b64encode(buffer).decode('utf-8')

            context = {
                'prediction_image': prediction,
                'count': count_objects,
                'majority_class': majority_class
            }

            return render(request, 'result.html', context)
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})
