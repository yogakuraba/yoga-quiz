#!pip install ultralytics
#!pip install deepface



from deepface import DeepFace
import cv2
import time
import os




results_list = []




models = ["VGG-Face","Facenet","Facenet512","OpenFace","SFace"]




for i in range(len(models)):
  
    try:
        result = DeepFace.verify(
            img1_path = "brad1.png",
            img2_path = "brad2.png",
            model_name = models[i]
        )

        result = result['verified']
        results_list.append(result)

    except Exception as e:
        #print(f"Model {models[i]} could not read the image. Check the image path or format like .jpg or .png")
        print(e)
    
final_result = max(results_list,key=results_list.count)





from IPython.display import display, Image
display(Image(filename="brad1.png"))
display(Image(filename="brad2.png"))
print(final_result)






# fig,ax = plt.subplots(1,2,figsize=(4,2))
# ax[0].imshow(plt.imread("brad1.png"))
# ax[1].imshow(plt.imread("brad2.png"))
# fig.suptitle(f"Verified {final_result}")
# plt.show()








face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

def capture_and_save_image(image_count):
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        
        file_name = f'captured_face_{image_count}.jpg'
        cv2.imwrite(file_name, frame)
        print(f"Image {image_count} saved as {file_name}")
    else:
        print(f"No faces detected in image {image_count}.")

for i in range(2):
    capture_and_save_image(i+1)
    time.sleep(5)

cap.release()








for i in range(len(models)):
  
    try:
        result = DeepFace.verify(
            img1_path = "captured_face_1.jpg",
            img2_path = "captured_face_2.jpg",
            model_name = models[i]
        )

        result = result['verified']
        results_list.append(result)

    except:
        print(f"Model {models[i]} could not read the image")
    
final_result = max(results_list,key=results_list.count)






display(Image(filename="captured_face_1.jpg"))
display(Image(filename="captured_face_2.jpg"))
print(final_result)






# fig,ax = plt.subplots(1,2,figsize=(4,2))

# ax[0].imshow(plt.imread("captured_face_1.jpg"))
# ax[1].imshow(plt.imread("captured_face_2.jpg"))
# fig.suptitle(f"Verified {final_result}")
# plt.show()








face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

def capture_and_save_image(image_count):
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        
        file_name = f'captured_face_{image_count}.jpg'
        cv2.imwrite(file_name, frame)
        print(f"Image {image_count} saved as {file_name}")
    else:
        print(f"No faces detected in image {image_count}.")

capture_and_save_image(1)
cap.release()

for i in range(len(models)):
  
    try:
        result = DeepFace.verify(
            img1_path = "captured_face_1.jpg",
            img2_path = "brad2.png",
            model_name = models[i]
        )

        result = result['verified']
        results_list.append(result)

    except:
        print(f"Model {models[i]} could not read the image. Check the image path or format like .jpg or .png")
    
final_result = max(results_list,key=results_list.count)









display(Image(filename="captured_face_1.jpg"))
display(Image(filename="brad1.png"))
print(final_result)





webcam = cv2.VideoCapture(0)

save_dir = "C:\\Users\\Abhiram Reddy\\Desktop\\face data\\captures"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

frame_count = 0
photo = "captured_face_1.jpg"

while True:
    ret, frame = webcam.read()
    if not ret:
        break 
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        
        face_file_path = os.path.join(save_dir, f"face_{frame_count}.png")
        cv2.imwrite(face_file_path, face)
            
        frame_count += 1
             
        try:
            result = DeepFace.verify(img1_path = face_file_path,
                                     img2_path = photo,
                                     model_name = models[0])
            
            prediction_label = str(result['verified']) + "    " + photo.split(".")[0]
            cv2.putText(frame, prediction_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error analyzing face: {e}")
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow("Output", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

webcam.release()
cv2.destroyAllWindows()








from ultralytics import YOLO




model = YOLO('yolov8m.pt')





results = model(source=0,show=True,conf=0.4,save=True)





video_path = ''
cap = cv2.VideoCapture(video_path)

ret = True

while ret:
    ret, frame = cap.read()
    if ret:
        results = model.track(frame, persist=True)
        frame_ = results[0].plot()
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break







FROM HERE:
CODE FOR BACKPACK AND PERSON
    

import cv2
import time

net = cv2.dnn.readNet("dnn_model/yolov4.weights", "dnn_model/yolov4.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Load class lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("Executing")

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


cv2.namedWindow("Frame")
c = 0
prev_x = 0

while True:

    ret, frame = cap.read()

    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=.4)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]

        if(prev_x and (x == prev_x or ((x < prev_x + 15) and (x > prev_x - 15))) and (y == prev_y or ((y < prev_y + 15) and (y > prev_y - 15)))):
            c += 1
            time.sleep(0.8)
        else:
            c = 0
        t = "Time" + "  :" +  str(c) + " secs"
        cv2.putText(frame, t, (x, y - 35), cv2.FONT_HERSHEY_PLAIN, 3, (200,0,50), 2)
        cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, (200,0,50), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200,0,50), 3)

    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        class_name = classes[class_id]

        if class_name == 'backpack':
            prev_x,prev_y,prev_w,prev_h = bbox
        else:
            prev_x = prev_y = 0

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()






FROM HERE:
CODE FOR COUSTUME VIDEO INPUT


import cv2
from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

model = YOLO('yolov8m.pt')

video_path = "C:\\Users\\Abhiram Reddy\\Downloads\\test.mp4"
cap = cv2.VideoCapture(video_path)

ret = True

while ret:
    ret, frame = cap.read()
    if ret:
        results = model.track(frame, persist=True)
        frame_ = results[0].plot()
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

































