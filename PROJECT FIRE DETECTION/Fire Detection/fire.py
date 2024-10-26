from ultralytics import YOLO
import cv2
import math
import winsound
import os
from dotenv import load_dotenv
from email.message import EmailMessage
import ssl
import smtplib
import time
import socket

load_dotenv()

email_sender = "tecmo.098@gmail.com"
password = os.getenv("PASSWORD")
email_reciver = "elvin.efren.soto@gmail.com"

subject = "Hemos detectado fuego"
body = """
     La camara de seguridad ha detectado una fuente sospechosa de fuego en el aula 08 de ISOF.
"""

em = EmailMessage()
em["From"] = email_sender
em["To"] = email_reciver
em["Subject"] = subject
em.set_content(body)

context = ssl.create_default_context()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

model = YOLO("fire.pt")

classnames = ['fuego']

last_email_time = 0
email_interval = 20

def check_internet():
    try:
        socket.create_connection(("www.google.com", 80))
        return True
    except OSError:
        pass
    return False

while True:
    ret, frame = cap.read()
    small_frame = cv2.resize(frame, (320, 240))
    result = model(small_frame, stream=True)

    fire_detected = False

    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 50:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1*2, y1*2), (x2*2, y2*2), (0, 0, 255), 5)
                cv2.putText(frame, f'{classnames[Class]} {confidence}%', (x1*2 + 8, y1*2 + 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

                fire_detected = True

                winsound.Beep(400, 1000)

    internet_connected = check_internet()

    if fire_detected and internet_connected and (time.time() - last_email_time) > email_interval:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.login(email_sender, password)
            smtp.sendmail(email_sender, email_reciver, em.as_string())
        last_email_time = time.time()
    elif fire_detected:
        winsound.Beep(400, 1000)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
