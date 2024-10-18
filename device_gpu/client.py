
import pickle
import socket
import struct
import cv2
import time
import argparse

class ImageStreamer:
    def __init__(self, host, port):
        self.host = host
        self.port = port

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Socket created')

        self.s.connect((self.host, self.port))
        print('Socket connected')

        self.payload_size = 0
        self.expected_msg_size = 0
        
        self.data = b''

    def get_next_frame(self):

        data = self.data
        # Calculate the payload size once
        payload_size = struct.calcsize("L")

        # Retrieve message size
        while len(data) < payload_size:
            data += self.s.recv(4096)

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0] ### CHANGED

        # Retrieve all data based on message size
        while len(data) < msg_size:
            data += self.s.recv(4096)

        frame_data = data[:msg_size]
        self.data = data[msg_size:]

        # Extract frame
        frame = pickle.loads(frame_data)
        return frame

    def close(self):
        self.s.close()

def main():
    parser = argparse.ArgumentParser(
        description="Image streaming client"
    )
    parser.add_argument("-ch", "--camera-host", type=str, default="pi-camera")
    parser.add_argument("-cp", "--camera-port", type=int, default=8089)
    args = parser.parse_args()

    image_streamer = ImageStreamer(args.camera_host, args.camera_port)
    while True:
        frame = image_streamer.get_next_frame()
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        time.sleep(2)

if __name__ == "__main__":
    main()