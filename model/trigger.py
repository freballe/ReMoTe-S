class Trigger:
    def __init__(self, category, origin, reason, new_building=None):   # origin is environment(1) or internal(2)
        self.category = category    # category is: opportunity, radical change, problem solving
        self.origin = origin
        self.reason = reason
        self.new_building = new_building