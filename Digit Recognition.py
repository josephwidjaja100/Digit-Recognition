# For live feed:

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

top, right, bottom, left = 10, 350, 250, 590

bg = None

def run_avg(image, aWeight):
    global bg

    if bg is None:
        bg = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
    global bg

    diff = cv2.absdiff(bg.astype("uint8"), image)

    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.IMREAD_GRAYSCALE)[1]

    (cnts, _) = cv2.findContours(thresholded.copy(),
                                 cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

    thresholded = cv2.cvtColor(thresholded, cv2.CV_32F)

    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

while True:
    # Read the frame
    _, img = cap.read()
    #img = cv2.flip(img, 1)
    roi = img[top:bottom, right:left]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    (thresholded, binary) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    digits = cv2.imread("digits.png", cv2.IMREAD_GRAYSCALE)

    rows = np.vsplit(digits, 50)
    cells = []
    for row in rows:
        rowCells = np.hsplit(row, 50)
        for cell in rowCells:
            cell = cell.flatten()
            cells.append(cell)
    cells = np.array(cells, dtype=np.float32)

    k = np.arange(10)
    labels = np.repeat(k, 250)

    testDigits = binary.astype(np.float32)
    testDigits = cv2.resize(testDigits, (20, 20))
    testDigits = np.vsplit(testDigits, 1)
    testCells = []
    for d in testDigits:
        d = d.flatten()
        testCells.append(d)
    testCells = np.array(testCells, dtype=np.float32)

    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

    knn = cv2.ml.KNearest_create()
    knn.train(cells, cv2.ml.ROW_SAMPLE, labels)
    ret, result, neighbours, dist = knn.findNearest(testCells, k=2)

    #cv2.putText(img, result, (350, 280), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,225), 2, cv2.LINE_AA)
    print(result)

    cv2.imshow('img', img)
    cv2.imshow('binary', binary)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==ord("q"):
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()

'''
If for non live feed:

import cv2
import numpy as np

digits = cv2.imread("digits.png", cv2.IMREAD_GRAYSCALE)
test_digits = cv2.imread("test_digits.png", cv2.IMREAD_GRAYSCALE)

rows = np.vsplit(digits, 50)
cells = []
for row in rows:
    row_cells = np.hsplit(row, 50)
    for cell in row_cells:
        cell = cell.flatten()
        cells.append(cell)
cells = np.array(cells, dtype=np.float32)
print("Cells: ")
print(cells)

k = np.arange(10)
cells_labels = np.repeat(k, 250)

test_digits = test_digits.astype(np.float32)
test_digits = cv2.resize(test_digits, (20, 20))
test_digits = np.vsplit(test_digits, 1)
test_cells = []
for d in test_digits:
    d = d.flatten()
    test_cells.append(d)
test_cells = np.array(test_cells, dtype=np.float32)
print("Test Cells: ")
print(test_cells)

# KNN
knn = cv2.ml.KNearest_create()
knn.train(cells, cv2.ml.ROW_SAMPLE, cells_labels)
ret, result, neighbours, dist = knn.findNearest(test_cells, k=3)


print(result)
'''
