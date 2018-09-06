from TileCoding import tiles, IHT


class MyCoding(object):

    def __init__(self, maxSize, numTilings, width, scale=[5, 6, 0.6, 7]):
        self.maxSize = maxSize
        self.numTilings = numTilings
        self.width = width
        self.iht = IHT(maxSize)
        self.scales = scale
        self.dim = len(scale)

    def mytiles(self, state):
        floats_list = [state[i] * self.width / self.scales[i] for i in range(self.dim)]
        coordinates = tiles(self.iht, self.numTilings, floats_list)
        return coordinates

class MyCoding3(object):

    def __init__(self, maxSize, numTilings, width, scale=[5, 6, 0.6, 7]):
        self.maxSize = maxSize
        self.numTilings = numTilings
        self.width = width
        self.iht = IHT(maxSize)
        self.scales = scale

    def mytiles(self, state):
        floats_list = [state[0] * self.width / self.scales[0], state[2] * self.width / self.scales[2], state[3] * self.width / self.scales[3]]
        coordinates = tiles(self.iht, self.numTilings, floats_list)
        return coordinates

class MyCoding2(object):

    def __init__(self, maxSize, numTilings, width, scale=[5, 6, 0.6, 7]):
        self.maxSize = maxSize
        self.numTilings = numTilings
        self.width = width
        self.iht = IHT(maxSize)
        self.scales = scale

    def mytiles(self, state):
        floats_list = [state[0] * self.width / self.scales[0], state[2] * self.width / self.scales[2]]
        coordinates = tiles(self.iht, self.numTilings, floats_list)
        return coordinates