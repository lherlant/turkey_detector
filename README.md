# Turkey Detector

Simple Python code to automatically detect when a turkey is visible on a usb camera, and then send an email alert with the image when turkeys are present. It is also trivial to change to detecting other classes of object instead of turkeys.

## Code organization

There are two folders in this repo:
- The folder `device_camera` is code that is meant to be run on a device with a USB camera attached. I used a Raspberry Pi 3 Model B Plus Rev 1.3 running the Bookworm version of Raspberry Pi OS, because that's what I had laying around. 
- The folder `device_gpu` is meant to run on a computer that has CUDA support and a decent GPU. I used an NVIDIA RTX 2080Ti on a machine running Ubuntu 22.04, because it's also what I had on hand, and wanted to keep everything local. I have two separate devices because the computer with the GPU is not located where I want the usb camera to be, and running a cable is too much of a pain.

## On Camera Device

Get the code:
```
git clone https://github.com/lherlant/turkey_detector.git
```

Make a virtualenv with python3 and install dependencies:
```
python3 -m venv turkeys
source turkeys/bin/activate
pip install -r device_camera/requirements.txt

```
Make sure the usb camera is plugged in, and note either the hostname or the IP address of the camera device. Then start the camera server:

```
python server.py
```
Note: If you want the server script to run after the terminal is killed, either add `&` at the end of the launch command or else use a tool like screen to start the process.

## On GPU Device

There is an assumption that the gpu device already has the required nvidia drivers and CUDA installed properly. If they are, running `nvidia-smi` should display a nice status screen with the available gpus and cuda version.

Get the code:
```
git clone https://github.com/lherlant/turkey_detector.git
```
Make a virtualenv with python3:
```
python3 -m venv turkeys
source turkeys/bin/activate
```

Install the appropriate version of [pytorch](https://pytorch.org/) for the installed cuda version. Mine is 11.8 so my command will be:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
``` 
Finally install remaining dependencies:
```
pip install -r device_gpu/requirements.txt
```

### Test the image streaming client

To test that the usb camera can connect and stream to the gpu device, run the follow with either the camera device's hostname or IP address:
```
python device_gpu/client.py --camera-host <CAMERA_DEVICE_HOSTNAME>
```
This will open an OpenCV window that displays the live stream. Note that only one connection can be made with the server at a time, so this should be closed before running the detection script. Press ESC to quit when the display window is in focus or Ctrl+C the script.

### Configure and test alerts

Now before the alerts can be launched, there are a few things that need to be configured. The alerts are sent via email. To do this, you must create a dedicated GMail account (don't recommend using credentials to an account used for  anything important). To allow the Python script to log in directly, you must create a 16-digit app password. Instructions for doing so can be found [here](https://support.google.com/accounts/answer/185833?hl=en). At the time of writing this option is only available after you enable 2-Step verification on the account. 

Once this is done, edit `device_gpu/config.py` to include the gmail username (without the `@gmail.com`) and the 16-digit app password. 

To test that the alerts are working, send a test alert to your chosen recipient address via:
```
python device_gpu/alert.py --recipients <YOUR-EMAIL@HERE.COM>
```
This should have sent a test email with attached test picture to your inbox. It may take a few minutes to be delivered. Check your spam filters.

TIP: Many cell phone providers have an email-to-text service. This means you can easily get near realtime alerts to your phone via email. Look up the email address format for your provider. Usually it's something like 10-digit-phonenumber@company-text.com

### Run the detector

With both the client and the alert features working, the detector can be run. The first time the detector script is run it will download modelweights from Huggingface, so will take a few minutes. 

```
python device_gpu/detect.py --camera-host <CAMERA_DEVICE_HOSTNAME> --recipients <YOUR-EMAIL@HERE.COM> --text "turkey"
```
This will launch a video preview window with the detections outlined in red bounding boxes (if present). Email alerts will be sent to recipients whenever a detection is made that matches one or more of the list of classes provided with the `--text` argument. There is some simple filtering in place such that an object must be detected for a few seconds before sending an alert (given by the value of `past_window_s`), and a second alert will not be sent for a least a certain number of minutes (given by the value of `min_s_before_realert`). Not every frame is processed, in order to reduce electricity use and heat generation by the GPU (change the value of `sleep_between_detections` to 0 to disable this behavior). 

The most recent image taken for each class will be stored locally where the detect script is run. Happy detecting!