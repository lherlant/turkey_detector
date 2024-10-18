import pickle
import socket
import struct

import cv2


cap=cv2.VideoCapture(0)

HOST = ''
PORT = 8089

while True:

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Socket created')

        s.bind((HOST, PORT))
        print('Socket bind complete')
        s.listen(10)
        print('Socket now listening')

        conn, addr = s.accept()

        data = b''
        payload_size = struct.calcsize("L")

        while True:
            # Get frame
            ret,frame=cap.read()
            print(f"Sending frame size {frame.shape}")
            # Serialize frame
            data = pickle.dumps(frame)

            # Send message length first
            message_size = struct.pack("L", len(data))

            # Then data
            conn.sendall(message_size + data)

        conn.close()
    except:
        pass