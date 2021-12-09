import cv2 as cv
import mss
import pyautogui as pag
import numpy as np


class ScreenManager:
    '''ScreenManager is designed to capture the game window and
    break the window into easily-accessible regions of interest.'''
    def __init__(self, screen_dims=None, regions=None):
        '''
        Arguments:
            screen_dims: a dictionary containing 'x', 'y' denoting
              the top-left corner of the game window and
              'h', 'w' denoting the window's height and width
            regions: a dictionary whose keys are the names of regions
              of interest and whose values are dictionaries containing
              'x', 'y', 'h', and 'w' for the ROI
        '''
        self._sct = mss.mss()
        self._screen_dims = screen_dims
        self._screen = None
        self.mouse_pos = None
        self.fast_mouse = False
        self._regions = regions

    def update(self):
        '''Grabs a screenshot of the screen and current mouse position.
        The awkard variables are to maximize the efficiency of
        pag.position() and _sct.grab(), since these must happen
        concurrently'''
        y = self._screen_dims['y']
        x = self._screen_dims['x']
        h = self._screen_dims['h']
        w = self._screen_dims['w']

        mouse_pos_i = pag.position()
        screen = np.array(
            self._sct.grab({'top': y,
                            'left': x,
                            'height': h,
                            'width': w}), dtype=np.uint8)
        mouse_pos_f = pag.position()

        '''
        m_x, m_y = mouse_pos_i
        if (m_x > x) and \
           (m_x < x+w) and \
           (m_y > y) and \
           (m_y < y+h):

            # local mouse coordinates relative to the captured screen
            screen = cv.circle(screen, (m_x-x, m_y-y), radius=1, color=(0, 0, 255), thickness=-1)

        m_x, m_y = mouse_pos_f
        if (m_x > x) and \
           (m_x < x+w) and \
           (m_y > y) and \
           (m_y < y+h):
            # local mouse coordinates relative to the captured screen
            screen = cv.circle(screen, (m_x-x, m_y-y), radius=1, color=(0, 255, 0), thickness=-1)
        '''
        self.mouse_pos = mouse_pos_i
        self._screen = screen

    def __getattr__(self, name):
        '''Allows the user to access ROIs'''
        r = self._regions[name]
        return self._screen[
            r['y']:r['y']+r['h'],
            r['x']:r['x']+r['w'], :-1]

    @property
    def screen(self):
        '''Getter for the whole screen'''
        return np.copy(self._screen[:, :, :-1])

    def annotated_screen(self):
        '''Returns the whole screen with ROIs outlined'''
        s = np.copy(self._screen)
        for key, r in self._regions.items():
            cv.rectangle(s,
                         (r['x'], r['y']),
                         (r['x']+r['w'], r['y']+r['h']),
                         (255, 0, 0),
                         2)
        # trim off alpha values
        return s[:, :, :-1]
