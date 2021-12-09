import timeit

import mss
import pyautogui as pag


def test_selective_screen_grab():

    setup = '''
import mss
import yaml

sct = mss.mss()

config = yaml.safe_load(open('screen_cfg.yaml'))

s_x = config['screen']['x'] + config['regions']['playscreen']['x']
s_y = config['screen']['y'] + config['regions']['playscreen']['y']
s_w = config['regions']['playscreen']['w']
s_h = config['regions']['playscreen']['h']
    '''

    stmt = '''
sct.grab({'top': s_y,
          'left': s_x,
          'height': s_h,
          'width': s_w})
    '''

    print(timeit.timeit(stmt=stmt, setup=setup, number=1000))


def test_whole_screen_grab():

    setup = '''
import mss
import yaml

sct = mss.mss()

monitor = sct.monitors[0]
    '''

    stmt = '''
sct.grab(monitor)
    '''

    print(timeit.timeit(stmt=stmt, setup=setup, number=1000))


if __name__ == '__main__':
    test_whole_screen_grab()
