# Driver-Drowsiness-Detection-using-YOLO
This project aims to detect driver drowsiness using YOLO (You Only Look Once) object detection framework. The model is trained to detect closed eyes, open eyes, yawns, and no yawns in images captured from the driver's camera.

![image](https://github.com/Aftabbs/Driver-Drowsiness-Detection-using-YOLO/assets/112916888/d0097428-1f91-4143-88a8-2527278217aa)

# Dataset
The dataset consists of two folders:
train_images: Contains images for training the model.
test_images: Contains images for evaluating the model's performance.

# Data Preprocessing
The images are resized to a standard size of 140x140 pixels to ensure consistency in the input size for the model. The images are then converted to RGB format as YOLO requires input images in RGB.
**Train Images**
![image](https://github.com/Aftabbs/Driver-Drowsiness-Detection-using-YOLO/assets/112916888/8f667c9c-9128-4cc2-8c15-d7190c43be43)

# Model Architecture
The YOLO object detection framework is used to train the model. YOLO is a real-time object detection system that can detect multiple objects in an image simultaneously. The model is trained to predict bounding boxes and class labels for the objects of interest (closed eyes, open eyes, yawns, and no yawns).

# Evaluation Metrics
The model's performance is evaluated using accuracy, precision, recall, and F1-score. The confusion matrix is also generated to visualize the model's predictions.

![image](https://github.com/Aftabbs/Driver-Drowsiness-Detection-using-YOLO/assets/112916888/2699ee90-d13b-4419-b562-1f77c46afa96)

# Results
The trained model achieved an accuracy of 98.6% on the test set. The precision and recall for each class are as follows:
![image](https://github.com/Aftabbs/Driver-Drowsiness-Detection-using-YOLO/assets/112916888/8ddcaaf8-8e75-4161-b540-ff914589fbd0)

**Predictions**
![image](https://github.com/Aftabbs/Driver-Drowsiness-Detection-using-YOLO/assets/112916888/a25f7a20-9b8c-4e2a-b556-517bbbe2f24d)

Sample images from the test set are visualized along with the model's predictions. The images are shown with the predicted class labels and bounding boxes.

# Conclusion
The YOLO model performs well in detecting driver drowsiness based on closed eyes, open eyes, yawns, and no yawns. The high accuracy and F1-scores indicate that the model can reliably detect driver drowsiness. However, further improvements and real-world testing are necessary before deploying the model in a practical setting.

# How to Run
Clone the repository.
Download the dataset and place it in the appropriate folders as mentioned above.
Open the driver_drowsiness_detection.ipynb notebook in Jupyter Notebook.
Run the notebook cells to train the model and evaluate its performance.

# Dependencies
The following libraries are required to run the code:
* numpy
* pandas
* matplotlib
* seaborn
* scikit-learn
* keras
* opencv-python
* glob
Make sure to install these dependencies before running the notebook.

# Credits
The YOLO implementation is based on the official YOLO repository: https://github.com/pjreddie/darknet

# License
This project is licensed under the MIT License. You are free to modify and use the code for your own projects.







