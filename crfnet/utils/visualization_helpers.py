"""
This script contains helper classes for visualizations
"""
# Standard Libraries
import threading
import queue

# 3rd Party Libraries
import cv2

# Local Libraries


class Cv2Threaded():
    _rendering_queue =  queue.Queue() # static queue for synchronizing all instances
    _keys_queue = queue.Queue() # static queue for synchronizing all instances
    _rendering_thread = None
    FPS = 24
    instance = None

    def __init__(self, fps=24, threaded=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if threaded and self.instance and self.instance.threaded:
            raise ValueError("Cv2Thread has already been started")

        self.threaded = threaded

    @classmethod
    def get_instance(cls, threaded=True, *args, **kwargs):
        if threaded:
            if cls.instance is None:
                cls.instance = Cv2Threaded(threaded=True, *args, **kwargs)
            
            return cls.instance
        else:
            return Cv2Threaded(threaded=False, *args, **kwargs)

    @classmethod
    def waitKey(cls, duration):
        key = -1
        try:
            while True:
                key = cls._keys_queue.get(block=False) # get all the keys, until no key is left
        except queue.Empty:
            pass
        return key

    @classmethod
    def __rendering_loop(cls):
        window_names = set()
        while cls._rendering_thread:
            try:
                winname, img = cls._rendering_queue.get(block=False)
                cv2.imshow(winname, img)
                window_names.add(winname)
            except queue.Empty:
                pass

            event_time = max(1000 // cls.FPS, 1)
            key = cv2.waitKey(event_time)
            if key != -1:
                cls._keys_queue.put(key)
        
        # finalize
        for winname in window_names:
            cv2.destroyWindow(winname)

    def imshow(self, winname, img):
        """
        Same usage as cv2.imshow
        """
        
        if self.threaded:
            cls = self.__class__
            if cls._rendering_thread is None:
                r_thread = threading.Thread(target=cls.__rendering_loop, daemon=True)
                cls._rendering_thread = r_thread
                cls._rendering_thread.start()

            cls._rendering_queue.put((winname, img))
        else:
            cv2.imshow(winname, img)

    def __del__(self):
        self.__class__._rendering_thread = None
    
    


