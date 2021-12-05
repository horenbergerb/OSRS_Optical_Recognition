from dataclasses import dataclass
from typing import Tuple, List

import cv2 as cv
import pyautogui as pag
import numpy as np

from src.utils.ScreenManager import ScreenManager


@dataclass
class Frame:
    img: np.ndarray
    mouse_pos: Tuple[int, int]
    # annotations formatted as [(top_left_coords), (w, h)]
    annotations: List[List[Tuple[int, int]]]


class VideoCapturer:
    '''
    Interactive tool to capture video of a screen along with
    supplementary information, such as mouse position.
    '''
    def __init__(self, screen_manager: ScreenManager, do_print=True):
        self._sm = screen_manager
        self._frames = []  # type: List[Frame]
        self.do_print = do_print

    def capture(self, delay=10):
        if self.do_print:
            print('Beginning video capture mode...')
            print('Press  \'c\' to start/stop capture, \'p\' to pause/play, '
                  'or \'q\' to terminate capture.')
        print('Capturing: False')
        capturing = False
        running = True
        while running:
            self._sm.update()
            s = self._sm.screen
            cv.imshow('Screen', s)

            if capturing:
                new_frame = Frame(self._sm.screen, pag.position(), [])
                self._frames.append(new_frame)

            key_pressed = cv.waitKey(delay)
            # pause
            if key_pressed == ord('p'):
                cv.imshow('Screen', s)
                cv.waitKey(0)

            # quit
            if key_pressed == ord('q'):
                if self.do_print:
                    print('Terminating capture mode...')
                return self._frames

            # capture
            if key_pressed == ord('c'):
                capturing = not capturing
                print('Capturing: {}'.format(capturing))


class VideoEditor:
    '''
    Tool for playback of video as well as
    adding box annotations to individual frames.
    '''
    def __init__(self, frames: List[Frame], do_print=True):
        self._idx = 0
        self._frames = frames
        self.do_print = do_print

    def _annotate_playback_image(self):
        cur_frame_img = np.copy(self._frames[self._idx].img)
        text = 'Frame {}/{}'.format(self._idx+1, len(self._frames))
        pos = (0, 30)
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)
        thickness = 1
        line_type = 2
        cv.putText(cur_frame_img, text, pos, font,
                   font_scale, font_color, thickness, line_type)
        for a in self._frames[self._idx].annotations:
            cv.rectangle(cur_frame_img, a[0], a[1], (255, 255, 255), 3)
        return cur_frame_img

    def playback(self, delay=10):
        if self.do_print:
            print('Beginning playback mode...')
            print('Press  \'p\' to pause/play, \',\' to step backward, '
                  '\'.\' to step forward, or \'q\' to terminate playback.')
            print('Click and drag on a paused frame to add annotation data.')

        # starts in pause loop
        running = True
        self._playback_pause_loop(delay)
        while running:
            cur_frame_img = self._annotate_playback_image()
            cv.imshow('Playback', cur_frame_img)
            # pause
            key_pressed = cv.waitKey(delay)
            if key_pressed == ord('p'):
                self._playback_pause_loop(delay)
            elif key_pressed == ord('q'):
                running = False

            if self._idx < len(self._frames)-1:
                self._idx += 1

    def _playback_on_mouse(self, event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            self.cur_box = [(x, y)]

        elif event == cv.EVENT_LBUTTONUP:
            self.cur_box = self.cur_box + [(x, y)]
            self._frames[self._idx].annotations.append(self.cur_box)

    def _playback_pause_loop(self, delay=10):
        cv.namedWindow('Playback')
        cv.setMouseCallback('Playback', self._playback_on_mouse, 0)
        while True:
            cur_frame_img = self._annotate_playback_image()
            cv.imshow('Playback', cur_frame_img)
            key_pressed = cv.waitKey(delay)
            if key_pressed == ord('p'):
                return
            if key_pressed == ord(',') and self._idx > 0:
                self._idx -= 1
            if key_pressed == ord('.') and self._idx < len(self._frames)-1:
                self._idx += 1
