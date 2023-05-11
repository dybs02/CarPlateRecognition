import re
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
        self.plate_aspect_ratio = 520.0 / 114.0
        self.plate_area = 18000
        self.text_pattern = re.compile(r'[A-Z0-9]{4,8}')

        self.CAPTURE.set(cv.CAP_PROP_FRAME_HEIGHT, 1920)
        self.CAPTURE.set(cv.CAP_PROP_FRAME_WIDTH, 1080)

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
            if plate_img is None:
                continue

            plate_number = self.retrive_plate_number(plate_img)


            if self.debug and plate_img is not None:
                cv.imshow('Plate image', plate_img)
                cv.imshow('Video capture', cv.resize(frame, (854, 480)))
                print(f'Found {plate_number = }')     


    def detectPlate(self, image):
        start_time = time.perf_counter()

        image = cv.resize(image, (854, 480))

        # Grayscale & noise reduction
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Disabled for better detection without plate frame
        # gray = cv.bilateralFilter(gray, 11, 17, 17)

        # Edge detection
        edged = cv.Canny(gray, 30, 200) 
        cnts = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:50]
        screenCnt = None

        for c in cnts:
            x, y, w, h = cv.boundingRect(c)
            aspect_ratio = float(w)/h
            if abs(aspect_ratio-self.plate_aspect_ratio) > 1.0 and w*h < self.plate_area:
                continue

            peri = cv.arcLength(c, closed=True)
            approx = cv.approxPolyDP(c, 0.05 * peri, closed=True)

            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is None:
            return None

        # Masking
        mask = np.zeros(gray.shape,np.uint8)
        new_image = cv.drawContours(mask, [screenCnt], 0, 255, -1)
        new_image = cv.bitwise_and(image, image, mask=mask)

        # Cropping
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        # Remove unnecessary plate part
        topy += int((bottomy-topy)*0.12)
        new_image =  gray[topx:bottomx+1, topy:bottomy+1]

        # Applying threshold
        _, new_image = cv.threshold(new_image, 50, 200, cv.THRESH_BINARY | cv.THRESH_OTSU)

        if self.debug:
            runtime = time.perf_counter() - start_time
            print(f'Plate detection {runtime = :.5f}s')
        
        return new_image


    def retrive_plate_number(self, image):
        start_time = time.perf_counter()

        text = pytesseract.image_to_string(image, config='--psm 11')
        text = text.replace(' ', '')
        match = re.search(self.text_pattern, text)

        if self.debug:
            runtime = time.perf_counter() - start_time
            print(f'Text detection  {runtime = :.5f}s')

        return match[0] if match is not None else ''