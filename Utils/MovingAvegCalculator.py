import math


class MovingAvegCalculator():
    def __init__(self, window_length):
        self.num_added = 0
        self.window_length = window_length
        self.window = [0.0 for _ in range(window_length)]

        self.aveg = 0.0
        self.var = 0.0

        self.last_std = 0.0

    def add_number(self, num):
        idx = self.num_added % self.window_length
        old_num = self.window[idx]
        self.window[idx] = num
        self.num_added += 1

        old_aveg = self.aveg
        if self.num_added <= self.window_length:
            delta = num - old_aveg
            self.aveg += delta / self.num_added
            self.var += delta * (num - self.aveg)
        else:
            delta = num - old_num
            self.aveg += delta / self.window_length
            self.var += delta * ((num - self.aveg) + (old_num - old_aveg))

        if self.num_added <= self.window_length:
            if self.num_added == 1:
                variance = 0.1
            else:
                variance = self.var / (self.num_added - 1)
        else:
            variance = self.var / self.window_length

        try:
            std = math.sqrt(variance)
            if math.isnan(std):
                std = 0.1
        except:
            std = 0.1

        self.last_std = std

        return self.aveg, std

    def get_standard_deviation(self):
        return self.last_std
