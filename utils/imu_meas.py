import numpy as np
class IMUmeas:
    def __init__(self, ts_, fib_b_, wib_b_, ba_, bg_):
        self.ts = ts_
        self.fib_b = fib_b_
        self.wib_b = wib_b_
        self.ba = ba_
        self.bg = bg_        