import cv2
import numpy as np
from keras.models import load_model


classifier = load_model('varshith.h5')

image_x, image_y = 64, 64


# Prediction
def predictor():
    from keras.preprocessing import image
    test_image = image.load_img('1.png', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)

    # Mapping the output to letters
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(26):
        if result[0][i] == 1:
            return letters[i]
    return ""


# cam predict
cam = cv2.VideoCapture(0)
img_text = ''

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)

    # region of interest
    img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2)
    imcrop = img[102:298, 427:623]

    # grayscale

    gray = cv2.cvtColor(imcrop, cv2.COLOR_BGR2GRAY)

    #gaussian blurr
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adaptive Thresholding
    mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # prediction text
    cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))


    cv2.imshow("Hand Gesture Recognition", frame)
    cv2.imshow("Processed Hand Mask", mask)

    # Save and predict the gesture when a frame is processed
    img_name = "1.png"
    save_img = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    img_text = predictor()


    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()