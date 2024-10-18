import argparse
import smtplib
from email.message import EmailMessage
from email.mime.image import MIMEImage
import atexit
from config import GMAIL_USERNAME, GMAIL_APP_PASSWORD
from pathlib import Path

class Alerter():
    def __init__(self, recipients):
        self.recipients = recipients
        self._authenticate()
        atexit.register(self.disconnect)
        self.new_message()
    
    def new_message(self):
        self.msg = EmailMessage()

    def _authenticate(self):
        self.smtp_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        self.smtp_server.login(GMAIL_USERNAME, GMAIL_APP_PASSWORD)

    def set_subject(self, subject):
        self.msg["Subject"] = subject

    def set_image(self, image_path):
        # Change message body to reference
        self.msg.set_content('''
        <!DOCTYPE html>
        <html>
            <body>
                    <p>Object Detection Automation</p>
                    <p><img src="cid:detections"/></p>
            </body>
        </html>
        ''', subtype='html')
        self.msg.make_mixed()

        # Then attach the image
        with open(image_path, 'rb') as f:
            msgImage = MIMEImage(f.read())
        msgImage.add_header('Content-ID', '<detections>')
        self.msg.attach(msgImage)

    def send_alert(self):
        self.msg["To"] = ", ".join(self.recipients)
        self.msg["From"] = f"{GMAIL_USERNAME}@gmail.com"
        try:
            self.smtp_server.send_message(self.msg)
        except (smtplib.SMTPServerDisconnected, smtplib.SMTPSenderRefused) as e:
            self._authenticate()
            self.smtp_server.send_message(self.msg)

    def disconnect(self):
        self.smtp_server.quit()

def main():
    parser = argparse.ArgumentParser(
        description="Email Alerter"
    )
    parser.add_argument("-r", "--recipients", nargs="+",
        required=True,
        help="email addresses to send alerts to. Separate multiple with spaces."    
    )
    args = parser.parse_args()

    alerter = Alerter(recipients = args.recipients)
    alerter.set_subject("Image test")
    test_file_abs_path = Path(__file__).parent / "test_turkey.png"
    alerter.set_image(test_file_abs_path)
    alerter.send_alert()

if __name__ == "__main__":
    main()