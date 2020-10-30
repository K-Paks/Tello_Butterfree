from easytello import tello
from easytello.stats import  Stats
import cv2
from threading import Thread
import threading


class PokeTello(tello.Tello):
    cap = None
    background_frame_read = None
    frame = None

    def send_command_no_response(self, command: str, query: bool = False):
        # New log entry created for the outbound command
        self.log.append(Stats(command, len(self.log)))

        # Sending command to Tello
        self.socket.sendto(command.encode('utf-8'), self.tello_address)
    def rc_control(self, a: int, b: int, c: int, d: int):
        self.send_command_no_response('rc {} {} {} {}'.format(a, b, c, d))

    def streamon(self):
        self.send_command('streamon')
        self.stream_state = True
        self.video_thread = threading.Thread(target=self._video_thread)
        self.video_thread.daemon = True
        self.video_thread.start()

    def _video_thread(self):
        # Creating stream capture object
        cap = cv2.VideoCapture('udp://' + self.tello_ip + ':11111')
        # Runs while 'stream_state' is True
        while self.stream_state:
            ret, frame = cap.read()
            # cv2.imshow('DJI Tello', frame)
            self.frame = frame