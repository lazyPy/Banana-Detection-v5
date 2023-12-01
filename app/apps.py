from django.apps import AppConfig
import tensorflow as tf
import random


class AppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app'

    def ready(self):
        # Load the TFLite model during application startup
        interpreter = tf.lite.Interpreter(model_path='static/quant.tflite')
        interpreter.allocate_tensors()

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
