from easytello import tello
from easytello.stats import  Stats

class PokeTello(tello.Tello):
    def send_command_no_response(self, command: str, query: bool = False):
        # New log entry created for the outbound command
        self.log.append(Stats(command, len(self.log)))

        # Sending command to Tello
        self.socket.sendto(command.encode('utf-8'), self.tello_address)
    def rc_control(self, a: int, b: int, c: int, d: int):
        self.send_command_no_response('rc {} {} {} {}'.format(a, b, c, d))

    def stop(self):
        self.send_command('stop')