class Cell :
    def __init__(self, neighbours , values  ):
        self.neighbours = neighbours;
        self.values= values;
        self.citysIndexs = []

    def cleanCitysInx(self):
        self.citysIndexs = []

    def addCityInx(self,i):
        self.citysIndexs.append(i)