import threading
import numpy as np

class AudioBuffer:
    def __init__(self):
        self.buffer = np.array([], dtype='int16')
        self.lock = threading.Lock()

    def add_chunk(self, chunk):
        with self.lock:
            self.buffer = np.concatenate((self.buffer, chunk))

    def get_and_clear(self, min_chunk, max_chunk=None):
        with self.lock:                
            if self.buffer.size >= min_chunk:
                if max_chunk and max_chunk > self.buffer.size:
                    data = self.buffer[:max_chunk].copy()
                    self.buffer = self.buffer[max_chunk:]
                else:
                    data = self.buffer.copy()
                    self.buffer = np.array([], dtype='int16')
                return data
            return None

    def size(self):
        with self.lock:
            return self.buffer.size