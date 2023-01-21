import cv2
import numpy as np
import tensorflow as tf
import time

#dataset keras kütüphanesinden yüklendi
mnist = tf.keras.datasets.mnist
#dataset test ve train olarak ayrıldı
(x_train,y_train), (x_test,y_test) = mnist.load_data()

#train ve test dataları normalize edildi (normalize etmenin ne olduğunu araştır, rapora yaz)
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#model oluşturuldu (modelin neden sequential olarak yapıldığını, sequential nedir, vs)
model = tf.keras.models.Sequential()

#model için farklı layerlar kuruldu (layer sayısı, neden flatten, dense, son layer neden 10luk, neden relu, son layer neden softmax)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) #probability layer

#model compile edildi (compile edilince ne olur)
#adam, scc, accuracy nedir neden kullanıldı
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy", metrics="accuracy")

#model fit edildi (fit edilince ne olur, epoch nedir, neden 5 epoch yapıldı)
model.fit(x_train,y_train,epochs=5)

#model.save metodu ile model kaydedildi
model.save('number_reader_model')

# Initialize the VideoCapture object (bu kısımdan sonrası farklı python dosyasında da olabilir)
#çünkü model her seferinde gereksizce yeniden eğitilir
capture = cv2.VideoCapture(0)

#model yüklenildi (farklı dosyada çalıştırıldığı farz edilerek)
model = tf.keras.models.load_model('number_reader_model')

while True:
    # Read a frame from the webcam
    _, frame = capture.read()

    # Preprocess the frame
    frame = cv2.resize(frame, (28, 28))  # Resize the frame to 28x28
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    frame_threshold = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]  # Apply thresholding
    # You may need to modify these preprocessing steps depending on your model and the way you have structured your data
    
    # Reshape the frame to (batch_size, rows, columns, channels)
    frame_reshaped = np.reshape(frame_threshold, (1, 28, 28, 1))
    
    # Use the model to predict the characters in the frame
    characters = model.predict(frame_reshaped)
    
    # Classify the characters
    # You will need to modify this step to match the output of your model
    classified_characters = []
    for character in characters:
        classified_characters.append(np.argmax(character))
    
    # Display the classified characters
    time.sleep(1)
    print(classified_characters)
    
    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Check for user input to stop the loop
    key = cv2.waitKey(5000) & 0xFF
    if key == ord('q'):
        break

# Release the VideoCapture object
capture.release()
