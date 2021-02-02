from time import sleep

def back_sleep(sec: int):
    sleep(sec)
    print('hey')

sec = 15
back_sleep(sec)
