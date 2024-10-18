from transformers import pipeline
import cv2
from PIL import Image 
import skimage
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import argparse
from alert import Alerter
from client import ImageStreamer
from datetime import datetime
from collections import defaultdict
import time

class ObjectDetector:
    
    def __init__(self, object_names, host=None, port=None, recipients=None, stream_device=None, send_alert=True, visualize=True):
        self.object_names = object_names
        self.remote_camera = False
        if stream_device is None:
            if host is None:
                self.stream_device = 0
            else:
                self.remote_camera = True
        else:
            self.stream_device = stream_device

        if recipients is None:
            recipients = []

        self.send_alert = send_alert
        self.visualize = visualize
        self.vis_window_name = "detections"
        if self.send_alert:
            self.alerter = Alerter(recipients)

        # Parameters to configure alert frequency
        self.min_s_before_realert = 60*10 # 10 mins = 60*10
        self.past_window_s = 3 # Must have been detected consistently for the past seconds
        self.score_threshold = 0.0 # generally if there's any detection it's pretty good
        self.last_detection_start = defaultdict(lambda: None)
        self.last_alert_time = defaultdict(lambda: None)
        for object_name in self.object_names:
            self.last_detection_start[object_name] = None
            self.last_alert_time[object_name] = None

        self.sleep_between_detections = 1

        # set up remote camera
        if self.remote_camera:
            self.image_streamer = ImageStreamer(host, port)

        # Download checkpoint and prepare detector
        checkpoint = "google/owlv2-base-patch16-ensemble"
        self.detector = pipeline(model=checkpoint, task="zero-shot-object-detection", device="cuda")

    def start_detections(self):
        if not self.remote_camera:
            # Start video capture device
            vc = cv2.VideoCapture(self.stream_device)

            # Try to get the first frame
            if vc.isOpened():
                rval, frame = vc.read()
            else:
                rval = False

            if self.visualize:
                self._start_vis_window(frame)

            # Loop over input frames
            while rval:
                rval, frame = vc.read()
                detections = self._get_detections(frame)
                if self.visualize:
                    annotated_frame = self._draw_detections(frame, detections)
                    self._render_frame(annotated_frame)
                    key = cv2.waitKey(20)
                    if key == 27: # exit on ESC
                        break

            # Clean up after
            vc.release()  

        else:
            viz_started = False
            while True:
                frame = self.image_streamer.get_next_frame()
                detections = self._get_detections(frame)
                if self.visualize:
                    if not viz_started:
                        self._start_vis_window(frame)
                        viz_started = True
                    annotated_frame = self._draw_detections(frame, detections)
                    self._render_frame(annotated_frame)
                    key = cv2.waitKey(20)
                    if key == 27: # exit on ESC
                        break
                    if self.send_alert:
                        self._alert_if_needed(detections, annotated_frame)
                # Let the GPU cool off...
                time.sleep(self.sleep_between_detections)

        if self.visualize:
            self._stop_vis_window()

    def _alert_if_needed(self, predictions, frame):
        """
        Only alert if there hasn't been an alert recently
        and if it's a consistent detection.
        """
        # Update last start times based on current predicitons
        detected_labels = []
        for prediction in predictions:
            label = prediction["label"]
            score = prediction["score"]
            if score > self.score_threshold:
                if self.last_detection_start[label] is None:
                    self.last_detection_start[label] = datetime.now()
                    print(f"Set start time for {label} to {self.last_detection_start[label]}")
                detected_labels.append(label)
        for label in self.last_detection_start.keys():
            if label not in detected_labels and self.last_detection_start[label] is not None:
                self.last_detection_start[label] = None
                print(f"Removing start time for {label} at {datetime.now()}")
        
        # Now decide if any alerts should be sent
        for label, start_detection_time in self.last_detection_start.items():
            need_to_send = False
            if start_detection_time is not None:
                s_detected = (datetime.now() - start_detection_time).seconds
                if s_detected >= self.past_window_s:
                    # We have detected long enough to want to send an alert
                    print(f"Could alert about {label}!")
                    # but need to check if we already sent an alert
                    if self.last_alert_time[label] is not None:
                        s_since_last_alert = (datetime.now() - self.last_alert_time[label]).seconds
                        if s_since_last_alert > self.min_s_before_realert:
                            need_to_send = True
                    else:
                        need_to_send = True
                    
                    if need_to_send:
                        print(f"SEND ALERT for {label}!!!")
                        self.last_alert_time[label] = datetime.now()
            if need_to_send:
                cv2.imwrite(f"{label}.png", frame)
                self.alerter.new_message()
                self.alerter.set_subject(f"{label} detected at {datetime.now()}")
                if label != "person":
                    self.alerter.set_image(f"{label}.png")
                self.alerter.send_alert()

    def _get_detections(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image).convert("RGB")
        predictions = self.detector(
            image,
            candidate_labels=self.object_names,
        )
        return predictions        

    def _start_vis_window(self, frame):
        cv2.namedWindow(self.vis_window_name)
        cv2.imshow(self.vis_window_name, frame)

    def _stop_vis_window(self):
        cv2.destroyWindow(self.vis_window_name)          

    def _draw_detections(self, frame, predictions):
        for prediction in predictions:
            box = prediction["box"]
            label = prediction["label"]
            score = prediction["score"]
            xmin, ymin, xmax, ymax = box.values()   
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
        return frame

    def _render_frame(self, frame):
        cv2.imshow(self.vis_window_name, frame)

def main():
    parser = argparse.ArgumentParser(
        description="Object detection streaming"
    )
    parser.add_argument("-f", "--filename", type=str,
        help="Video filename if want to load from a video. If not provided will stream from remote camera client."
    )
    parser.add_argument("-ch", "--camera-host", type=str, default="pi-camera")
    parser.add_argument("-cp", "--camera-port", type=int, default=8089)
    parser.add_argument("-r", "--recipients", nargs="+",
        required=True,
        help="email addresses to send alerts to. Separate multiple with spaces."    
    )
    parser.add_argument("-t", "--text", type=str,
        nargs="+", default=["turkey"],
        help="List of object text classes to match. Separate multiple with spaces. Default ['turkey']"
    )
    args = parser.parse_args()

    obj_detector = ObjectDetector(
        args.text,
        stream_device = args.filename,
        host = args.camera_host,
        port = args.camera_port,
        send_alert = True,
        visualize = True,
        recipients = args.recipients
    )
    obj_detector.start_detections()


if __name__ == "__main__":
    main()
