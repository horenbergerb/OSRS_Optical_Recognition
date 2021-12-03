import cv2 as cv
import mss
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
        self._regions = regions

    def update(self):
        '''Grabs a screenshot of the screen'''
        self._screen = np.array(
            self._sct.grab({'top': self._screen_dims['y'],
                            'left': self._screen_dims['x'],
                            'height': self._screen_dims['h'],
                            'width': self._screen_dims['w']}), dtype=np.uint8)

    def __getattr__(self, name):
        '''Allows the user to access ROIs'''
        r = self._regions[name]
        return self._screen[
            r['y']:r['y']+r['h'],
            r['x']:r['x']+r['w'], :-1]

    @property
    def screen(self):
        '''Getter for the whole screen'''
        return np.copy(self._screen)

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
