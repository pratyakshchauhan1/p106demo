import cv2


# Create our body classifier
vid = cv2.VideoCapture(0) 
fc = cv2.CascadeClassifier("s.xml")
body_classifier = cv2.CascadeClassifier('1.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    ret, frame = vid.read()
    greyimage = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
    faces = fc.detectMultiScale(greyimage) 
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(greyimage,1.2, 3)
    
    # Extract bounding boxes for any bodies identified
    

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
