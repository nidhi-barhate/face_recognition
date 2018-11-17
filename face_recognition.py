import cv2
import numpy as np
import os

LBP_faceCascade = cv2.CascadeClassifier('cascades/data/lbpcascade_frontalface.xml')

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = LBP_faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    #extract the face area
    (x, y, w, h) = faces[0]
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

def preparing_traning_data(folder_path):
    dirs = os.listdir(folder_path)
    
    faces = []
    labels = []
    Ids = []
    id_n = 0
    
    for dir_name in dirs:
        label = dir_name
        sub_dir_path = folder_path + "/" + dir_name
        sub_images_names = os.listdir(sub_dir_path)
        id_n = id_n+1
        
        for image_name in sub_images_names:
            
            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
            
            image_path = sub_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)
            
            #detect face
            face, rect = detect_face(image)
            if face is not None:
                faces.append(face)
                labels.append(label)
                Ids.append(id_n)
        
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels, Ids


print("Preparing data...")
faces, labels, Ids = preparing_traning_data("dataset")
print("Data prepared")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))


LBPHFaceRecognizer = cv2.face.LBPHFaceRecognizer_create()

#train our face recognizer of our training faces
LBPHFaceRecognizer.train(faces, np.array(Ids))
LBPHFaceRecognizer.save('LBPHFaceRecognizer.yml')

face_recognizer = LBPHFaceRecognizer

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)

    #predict the image using our face recognizer 
    label, confidence = face_recognizer.predict(face)
    
    #get name of respective label returned by face recognizer
    label_text = subjects[label]
    
    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    #draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img,label,confidence

###############################################
subjects = ['None']
subjects = subjects + os.listdir('dataset')
print("Predicting images...")

#load test images
test_img1 = cv2.imread("examples/JB.jpg")
test_img2 = cv2.imread("examples/SM.jpg")

#perform a prediction
predicted_img1,pridected1,confidence1 = predict(test_img1)
predicted_img2,pridected2,confidence2 = predict(test_img2)
print("Prediction complete")

#display both images
cv2.imshow(subjects[pridected1], cv2.resize(predicted_img1, (400, 500)))
cv2.imshow(subjects[pridected2], cv2.resize(predicted_img2, (400, 500)))

 #save image
cv2.imwrite('output/'+subjects[pridected1]+'.jpg', predicted_img1)
cv2.imwrite('output/'+subjects[pridected2]+'.jpg', predicted_img2)

print(confidence1)
print(confidence2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()