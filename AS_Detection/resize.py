# import os
# import cv2

# input_dir = r"C:\Users\venkatesh\Desktop\pro\dataset"
# output_dir = "resized_dataset"
# target_size = (224, 224)

# os.makedirs(output_dir, exist_ok=True)

# for label in os.listdir(input_dir):
#     class_dir = os.path.join(input_dir, label)
#     save_dir = os.path.join(output_dir, label)
#     os.makedirs(save_dir, exist_ok=True)

#     for img_name in os.listdir(class_dir):
#         img_path = os.path.join(class_dir, img_name)
#         image = cv2.imread(img_path)
#         if image is None:
#             continue
#         resized = cv2.resize(image, target_size)
#         cv2.imwrite(os.path.join(save_dir, img_name), resized)

# print("✅ Resizing complete. Saved in 'resized_dataset'")
import os, shutil
from sklearn.model_selection import train_test_split

source_dir = 'resized_dataset'
output_dir = 'resized_dataset_split'
split_ratio = 0.2  # 20% for validation

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    images = os.listdir(class_path)

    train_imgs, val_imgs = train_test_split(images, test_size=split_ratio, random_state=42)

    for phase, img_list in zip(['train', 'val'], [train_imgs, val_imgs]):
        save_dir = os.path.join(output_dir, phase, class_name)
        os.makedirs(save_dir, exist_ok=True)

        for img in img_list:
            src = os.path.join(class_path, img)
            dst = os.path.join(save_dir, img)
            shutil.copy(src, dst)

print("✅ Dataset split into train/val successfully.")
