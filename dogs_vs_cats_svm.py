import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = []
labels = []

base_path = "kagglecatsanddogs_5340/PetImages"

for category in ["Cat", "Dog"]:
    folder = os.path.join(base_path, category)
    label = 0 if category == "Cat" else 1

    for file in os.listdir(folder)[:200]:
        try:
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (64, 64))

            data.append(img.flatten())
            labels.append(label)

        except:
            pass

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = SVC(kernel="linear")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))