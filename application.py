import cv2
import os
import numpy as np
import tensorflow as tf
import keras


#Chargement du modèle permettant de détecter le port du masque
modelMasque = keras.models.load_model("./modele/mod.h5")

#Capture de la caméra (idCamera)
cap = cv2.VideoCapture(0)

while True:
   
    _, image = cap.read()

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    

    #Appel du modèle appris pour la detection de masque
    capture = cv2.resize(image, (224, 224))
    capture = capture.reshape((1, capture.shape[0], capture.shape[1], capture.shape[2]))
    predict = modelMasque.predict(capture)
    print(predict)
    pasDeMasque = predict[0][0]
    avecMasque = predict[0][1]

    # Interpretation de la prediction
    if (pasDeMasque > avecMasque):
        # cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
        cv2.putText(image, "PAS DE MASQUE", (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    else:
        # cv2.rectangle(image, (startX, startY), (endX, endY),(0, 255, 0), 2)
        cv2.putText(image, "OK", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)


    # Affichage de l'image
    cv2.imshow('img', image)
    
    # touche de sortie du programme
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()
