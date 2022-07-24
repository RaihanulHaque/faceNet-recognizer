from time import sleep
from pyfirmata import Arduino, SERVO

board = Arduino("/dev/ttyACM0")
pin = 10
board.digital[pin].mode = SERVO
INITIALANGLE = 0

# def rotateServo(pin, angle):
#     board.digital[pin].write(angle)
#     sleep(0.015)
# while True:


def unlock():
    board.digital[pin].write(90)
    sleep(1)
    return 


def lock():
    board.digital[pin].write(0)
    sleep(1)


lock()
