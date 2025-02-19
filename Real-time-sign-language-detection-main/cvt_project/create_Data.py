import cv2
import time
import numpy as np
import os

image_x, image_y = 64, 64

#trackbar
def nothing(x):
    pass


def create_folder(folder_name):
    if not os.path.exists('C:/Users/lenovo/PycharmProjects/pythonProject2/newhand/training_set/' + folder_name):
        os.mkdir('C:/Users/lenovo/PycharmProjects/pythonProject2/newhand/training_set/' + folder_name)
    if not os.path.exists('C:/Users/lenovo/PycharmProjects/pythonProject2/newhand/test_set/'+ folder_name):
        os.mkdir('C:/Users/lenovo/PycharmProjects/pythonProject2/newhand/test_set/' + folder_name)

#capturing images
def capture_images(ges_name):
    create_folder(str(ges_name))

    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")

    img_counter = 0
    t_counter = 1
    training_set_image_name = 1
    test_set_image_name = 1

    for _ in range(5):
        while True:
            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)

            # region of interest for hand capture
            img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2)
            imcrop = img[102:298, 427:623]

            #  grayscale conversion
            gray = cv2.cvtColor(imcrop, cv2.COLOR_BGR2GRAY)

            # gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)

            # adaptive thresholding for brightness adjustment
            mask = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            cv2.putText(frame, str(img_counter), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
            cv2.imshow("test", frame)
            cv2.imshow("mask", mask)

            if cv2.waitKey(1) == ord('x'):
                if t_counter <= 400:
                    img_name = 'C:/Users/lenovo/PycharmProjects/pythonProject2/newhand/training_set/' + str(ges_name) + "/{}.png".format(
                        training_set_image_name)
                    save_img = cv2.resize(mask, (image_x, image_y))
                    cv2.imwrite(img_name, save_img)
                    print("{} written!".format(img_name))
                    training_set_image_name += 1

                if t_counter > 400 and t_counter <= 500:
                    img_name = 'C:/Users/lenovo/PycharmProjects/pythonProject2/newhand/test_set/' + str(ges_name) + "/{}.png".format(test_set_image_name)
                    save_img = cv2.resize(mask, (image_x, image_y))
                    cv2.imwrite(img_name, save_img)
                    print("{} written!".format(img_name))
                    test_set_image_name += 1
                    if test_set_image_name > 100:
                        break

                t_counter += 1
                if t_counter == 501:
                    t_counter = 1
                img_counter += 1

            elif cv2.waitKey(1) == 27:
                break

        if test_set_image_name > 100:
            break

    cam.release()
    cv2.destroyAllWindows()


ges_name = input("Enter gesture name: ")
capture_images(ges_name)