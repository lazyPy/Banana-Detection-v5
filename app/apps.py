from django.apps import AppConfig
import tensorflow as tf
import random
from django.conf import settings


class AppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app'

    def ready(self):
        bucket_name = settings.AWS_STORAGE_BUCKET_NAME
        file_name = 'quant.tflite'

        # Initialize S3 client
        s3_client = settings.S3_CLIENT

        # Load the TFLite model from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=file_name)
        tflite_model_content = response['Body'].read()
        interpreter = tf.lite.Interpreter(model_content=tflite_model_content)


        # Load the TFLite model during application startup
        # interpreter = tf.lite.Interpreter(model_path='static/quant.tflite')
        # interpreter.allocate_tensors()

        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Names of the classes according to class indices
        names = ['Bungulan', 'Cardava', 'Lacatan']

        # Creating random colors for bounding box visualization
        colors = {name: [random.randint(0, 255) for _ in range(3)] for name in names}

        # Save the initialized elements into the app configuration for access later
        self.interpreter = interpreter
        self.input_details = input_details
        self.output_details = output_details
        self.names = names
        self.colors = colors
