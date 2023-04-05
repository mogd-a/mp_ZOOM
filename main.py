import cv2
import mediapipe
import pyvirtualcam

# definirea variabilelor
mp_face_detection = mediapipe.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()
cap = cv2.VideoCapture(0)
zoom_level = 1.0
smoothing_factor = 0.9
prev_cx = None
prev_cy = None
ena = True

def on_zoom_trackbar(val):
    global zoom_level
    zoom_level = 3.0 - val / 50

def on_smoothing_trackbar(val):
    global smoothing_factor
    smoothing_factor = val / 100

cv2.namedWindow('mp_ZOOM', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Zoom', 'mp_ZOOM', 0, 100, on_zoom_trackbar)
cv2.createTrackbar('Smoothing', 'mp_ZOOM', 90, 100, on_smoothing_trackbar)

with pyvirtualcam.Camera(width=640, height=360, fps=30) as cam:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Conversie RGB-BGR ca așa merge mai bine la face detection.
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = face_detection.process(bgr_frame)

        # gasim punctele după care ne orientăm
        if results.detections and ena:
            for detection in results.detections:
                x1 = int(detection.location_data.relative_bounding_box.xmin * frame.shape[1])
                y1 = int(detection.location_data.relative_bounding_box.ymin * frame.shape[0])
                x2 = int(x1 + detection.location_data.relative_bounding_box.width * frame.shape[1])
                y2 = int(y1 + detection.location_data.relative_bounding_box.height * frame.shape[0])
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w // 2
                cy = y1 + h // 2

                if prev_cx is not None and prev_cy is not None:
                    cx = int(smoothing_factor * prev_cx + (1 - smoothing_factor) * cx)
                    cy = int(smoothing_factor * prev_cy + (1 - smoothing_factor) * cy)

                prev_cx = cx
                prev_cy = cy

                w = int(w * zoom_level)
                h = int(h * zoom_level)
                x1 = max(0, cx - w // 2)
                y1 = max(0, cy - h // 2)
                x2 = min(frame.shape[1], cx + w // 2)
                y2 = min(frame.shape[0], cy + h // 2)
                cropped_frame = frame[y1:y2, x1:x2]
                resized_frame = cv2.resize(cropped_frame, (640, 360))
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
                cam.send(rgb_frame)
                cam.sleep_until_next_frame()
        #daca nu detecteaza vreo față arată tot
        else: 
            cam.send(cv2.cvtColor(cv2.resize(frame, (640, 360)),cv2.COLOR_BGR2RGB))
        key = cv2.waitKey(1)
        #programul se inchide dacă e apasat „q” sau e închisă fereastra
        if key == ord('q') or cv2.getWindowProperty('mp_ZOOM', cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key == ord('z'):
            ena = not ena
        cv2.imshow('mp_ZOOM', frame)
         
cv2.destroyAllWindows()