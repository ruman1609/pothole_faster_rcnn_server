from django.db import models
import datetime

def image_upload_to(instance, name):
    time = datetime.datetime.now()
    time = time.strftime("%Y_%m_%d")
    return f"images/{time}_{name}"

# Create your models here.

class Prediction(models.Model):
    id = models.BigAutoField(primary_key=True)
    image = models.ImageField(upload_to=image_upload_to,
                              default="images/default.jpg")
