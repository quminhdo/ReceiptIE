class Box:

    x1 = None
    x2 = None
    y1 = None
    y2 = None

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def xc(self):
        return int((self.x2 + self.x1) / 2)

    @property
    def yc(self):
        return int((self.y2 + self.y1) / 2)
    
    @property
    def w(self):
        return self.x2 - self.x1
    
    @property
    def h(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.w * self.h

class Component(Box):

    def __init__(self, 
                text,
                x1, 
                y1, 
                x2, 
                y2, 
                page_width, 
                page_height):
        super(Component, self).__init__(x1, y1, x2, y2)
        self.text = text
        self.pw = page_width
        self.ph = page_height

    @property
    def norm_xc(self):
        return self.xc/self.pw

    @property
    def norm_yc(self):
        return self.yc/self.ph

    @property
    def position(self):
        raise NotImplementedError

def get_intersection_box(b1, b2):
    if b1.x1 > b2.x2 or b2.x1 > b1.x2:
        return Box(0, 0, 0, 0)
    if b1.y1 > b2.y2 or b2.y1 > b2.y2:
        return Box(0, 0, 0, 0)
    x1 = max(b1.x1, b2.x1)
    y1 = max(b1.y1, b2.y1)
    x2 = min(b1.x2, b2.x2)
    y2 = min(b1.y2, b2.y2)
    return Box(x1, y1, x2, y2)