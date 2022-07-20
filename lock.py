from pyfirmata import Arduino, SERVO

board = Arduino("/dev/ttyACM0")
board.digital[9].mode = SERVO
INITIALANGLE = 0

while True:
    board.digital[9].write(0)