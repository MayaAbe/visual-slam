import cv2

# 画像の読み込み
img = cv2.imread('image.jpg')

# 物体検出器の読み込み
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 物体検出の実行
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

# 検出された物体を可視化する
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# 結果を表示する
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
