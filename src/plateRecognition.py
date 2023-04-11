import time

import cv2 as cv
import imutils
import numpy as np
import pytesseract


class PlateRecognition():

    def __init__(self, debug = False):
        self.CAPTURE = cv.VideoCapture(0)
        self.debug = debug
        self.frame_rate = 20

        if not self.CAPTURE.isOpened():
            print("Cannot open camera")
            exit()

    def run(self):
        prev = 0

        while True:
            if cv.waitKey(1) == ord('q'):
                self.CAPTURE.release()
                cv.destroyAllWindows()
                return

            # framerate control
            if time.time()-prev <= 1.0/self.frame_rate:
                continue
            prev = time.time()

            # Plate detection
            ret, frame = self.CAPTURE.read()
            plate_img = self.detectPlate(frame)

            if self.debug and plate_img is not None:
                cv.imshow('Video capture', plate_img)  


    def detectPlate(self, image):
        start_time = time.perf_counter()

        image = cv.resize(image, (854, 480))

        # Grayscale & noise reduction
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = cv.bilateralFilter(gray, 11, 17, 17)

        # Edge detection
        edged = cv.Canny(gray, 30, 200) 
        cnts = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:20]
        screenCnt = None

        for c in cnts:
            peri = cv.arcLength(c, closed=True)
            approx = cv.approxPolyDP(c, 0.018 * peri, closed=True)

            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is None:
            return None

        # Masking
        mask = np.zeros(gray.shape,np.uint8)
        new_image = cv.drawContours(mask, [screenCnt], 0, 255, -1)
        new_image = cv.bitwise_and(image, image, mask=mask)

        if self.debug:
            runtime = time.perf_counter() - start_time
            print(f'Plate detection {runtime = }')
            print(f'Time to next frame = {(1.0/self.frame_rate) - runtime}')
        
        return new_image


    def retrive_plate_number(self, image):
        pass