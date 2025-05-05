import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


val_dir = 'resized_dataset_split/val'
img_size = (224, 224)
batch_size = 32
# Data loader
val_datagen = ImageDataGenerator(rescale=1./255)
val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Load models

mobilenet_model = load_model("mobilenetv2_balanced_model.h5")
vgg16_model = load_model("V:\Projects\Projects\IBM - 2\vgg16_balanced_model.h5")

# Evaluate models

models = {
    "MobileNetV2": mobilenet_model,
    "VGG16": vgg16_model
}

results = {}

for name, model in models.items():
    print(f"🔍 Evaluating {name}...")
    eval_result = model.evaluate(val_data, verbose=1)
    results[name] = {
        "loss": eval_result[0],
        "accuracy": eval_result[1]
    }

# Plot comparison

def plot_model_comparison(results, metric='accuracy'):
    plt.figure(figsize=(6, 5))
    model_names = list(results.keys())
    values = [results[name][metric] for name in model_names]
    plt.bar(model_names, values, color=['skyblue', 'orange'])
    plt.title(f'{metric.title()} Comparison')
    plt.ylabel(metric.title())
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{metric}_comparison_bar.png")
    plt.show()

# Plot accuracy comparison

plot_model_comparison(results, metric='accuracy')

# Plot loss comparison
plot_model_comparison(results, metric='loss')


