import keyboard
import time

def key_press(key):
    print(key.name)

keyboard.on_press(key_press)

while True:
    time.sleep(1)
