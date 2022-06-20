import copy
import sys

from random import random, randint
from typing import *
import pygame
from Cell import *
from scipy.spatial import distance
import matplotlib.cm
#learning rate
alpha=0.5
#distance influence
teta=[0.3,0.2,0.1]
#input
inputSet: List[List[float]] = []
#city names
cityName: List[str]= []
#city ecomonic level
cityEcomonic: List[int] = []
#weights
somMap: List[List[Cell]]= []
#iteration number
M = 10
#input vector length. Or number of parties
L = 13
#the size X size of the somMap grid
SizeSom = 9


def readInput(path):
    '''
    read the input file. For city i (0<=i<=N) save her name, her economic state,
    vector of voting percentage
    :param path: file path 
    :return: None
    '''
    global cityName, cityEcomonic, inputSet,L
    data = open(path, 'r')
    for i,line in enumerate(data):
        line = line.split(",")
        if i==0:
            continue
        cityName.append(line[0])
        cityEcomonic.append(int(line[1]))
        totalVotes = float(line[2])
        line2 = line[3:-1]
        if line[-1][-1] == "\n":
            line2.append(line[-1][:-1])
        else:
            line2.append(line[-1])
        v = []
        for n in line2:
            v.append(float(n)/totalVotes)
        inputSet.append(v)
    L = len(inputSet[0])




def randomVector():
    '''
    :return: vector of len L, with random values between 0 to 1
    '''
    global L
    v = []
    for i in range(L):
        v.append(random())
    return v
    # return inputSet[randint(0,len(inputSet)-1)]

def initialize(path):
    '''
    initialize the net.
    the net is made of 61 haxgons but we save them in 9x9 grid, where 20 values
    out of the 81, are Nones
    :return: None
    '''
    global somMap, SizeSom
    readInput(path)
    numberOfHexagonsInRow = 5
    for i in range(SizeSom):
        currSomMapRow: List[Cell] = []
        for j in range(SizeSom):
            # if shift[i] <= j < shift[i]+numberOfHexagonsInRow :
            if j < numberOfHexagonsInRow:

                currSomMapRow.append(Cell([], randomVector()))
            else:
                currSomMapRow.append(None)
        if i < (SizeSom - 1) / 2:
            numberOfHexagonsInRow += 1
        else:
            numberOfHexagonsInRow -= 1
        somMap.append(currSomMapRow)
    # to each cell, set her Neighbours
    for i, row in enumerate(somMap):
        for j, cell in enumerate(row):
            if cell != None:
                cell.neighbours = getNeighbours(i, j)
    print("done init")

def getNeighbours(i,j):
    '''
    :param i: the row index
    :param j: the column index
    :return: the list of Neighbours indexs, of that haxgon 
    '''
    neb = []
    
    # Neighbours in the same row
    if 0<j and somMap[i][j-1] != None:
        neb.append((i,j-1))
    if j+1 < SizeSom and somMap[i][j+1] != None:
        neb.append((i,j+1))
    
    #Neighbours from the row above
    if 0<i:
        if somMap[i-1][j] != None :
            neb.append((i-1, j ))

        if j+1 < SizeSom  and 4<i and somMap[i-1][j+1] != None:
            neb.append((i-1, j + 1))
        if 0<=j-1 and i<=4 and somMap[i-1][j-1] != None :
            neb.append((i-1, j - 1))

    #Neighbours from the row underneth
    if i<SizeSom-1:
        if somMap[i + 1][j] != None:
            neb.append((i+1, j))

        if 0<= j - 1 and 4 <= i and somMap[i+1][j-1] != None :
            neb.append((i + 1, j - 1))
        if j + 1 < SizeSom and i<4 and somMap[i+1][j+1] != None :
            neb.append((i + 1, j + 1))
    return neb





def findBmu(indexInput):
    '''

    :param indexInput: the index of the current city vector
    :return: the coordinates of the haxgon that has the vector that is the closet (by euclidean distance)
     to the city vector.
    '''
    global somMap,inputSet
    minDistance=float('inf')
    bmuCoordinates= (0,0)
    for i,row in enumerate(somMap):
        for j,cell in enumerate(row):
            if(cell!=None):
                d=distance.euclidean(cell.values,inputSet[indexInput])
                if(minDistance>d):
                    minDistance=d
                    bmuCoordinates= (i,j)
    return bmuCoordinates


def updateWeight(indexInput , coordinatesSom , distanceLevel):
    '''
    update on single neuron
    :param indexInput: the city vector index
    :param coordinatesSom: the haxgon index
    :param distanceLevel: distanceLevel
    :return: the haxgon new vector
    '''
    result=[]
    for i in range(len(inputSet[indexInput])):
        (x,y)=coordinatesSom
        ui=somMap[x][y].values[i]
        vi=inputSet[indexInput][i]
        result.append(ui+alpha*teta[distanceLevel]*(vi-ui))
    return result

def cleanCitysInx():
    '''
    From each cell in the map, clean the cities that were positioned in the cells
    :return: None
    '''
    for row in somMap:
        for cell in row:
            if cell != None:
                cell.cleanCitysInx()

def update(copySomMap,indexInput , coordinatesSom):
    '''
    :param copySomMap: copy of the map
    :param indexInput: the current city vector
    :param coordinatesSom: the current haxgon index
    :return: the copy after changes
    '''
    global somMap
    #update all neighbours and bmv
    front = []
    #update bmv
    (x,y)=coordinatesSom
    copySomMap[x][y].addCityInx(indexInput)
    copySomMap[x][y].values=updateWeight(indexInput,coordinatesSom,0)
    front.append(coordinatesSom)
    #first neighbours
    for neighbourCoordinates in somMap[x][y].neighbours:
        if(neighbourCoordinates not in front):
            (i,j)=neighbourCoordinates
            if somMap[i][j] == None:
                continue
            copySomMap[i][j].values=updateWeight(indexInput,neighbourCoordinates,1)
            front.append(neighbourCoordinates)
    # second neighbours
    for secondNeighboursCoordinates in front:
        (x,y)=secondNeighboursCoordinates
        if somMap[x][y] == None:
            continue
        for neighbourCoordinates in somMap[x][y].neighbours:
            if (neighbourCoordinates not in front):
                (i, j) = neighbourCoordinates
                if somMap[i][j] == None:
                    continue
                copySomMap[i][j].values = updateWeight(indexInput, neighbourCoordinates, 2)
                front.append(neighbourCoordinates)
    return copySomMap

def epoch():
    '''
    run one single epoch
    :return: None
    '''
    global somMap,inputSet
    cleanCitysInx()
    # copySomMap = copy.deepcopy(somMap)
    copySomMap = somMap
    for inputInx in range(len(inputSet)):
        haxInx = findBmu(inputInx)
        copySomMap = update(copySomMap,inputInx,haxInx)
    somMap = copySomMap


import matplotlib
cmap = matplotlib.cm.ScalarMappable(norm = matplotlib.colors.Normalize(vmin=1, vmax=10,clip=True))
def chooseColorForHax(economicLevel):
    '''

    :param economicLevel: val between 1 to 10
    :return: color that were choosed by the economicLevel
    '''
    if economicLevel == -1:
        return (128,128,128)
    # return  (math.ceil(economicLevel*25.5),math.ceil(economicLevel*10.5),math.ceil(economicLevel*15.5))
    color_0_1 = cmap.to_rgba(economicLevel)
    color_0_255 = []
    for c in color_0_1:
        color_0_255.append(c*255)
    return tuple(color_0_255)

def getEcomonicAvg(i,j):
    '''
    :param i: row
    :param j: column
    :return: the average economic state of the cities that were allocated to that cell
    '''
    global cityEcomonic, somMap
    s = 0
    if somMap[i][j] == None :
        return -1
    cellCitysInxs = somMap[i][j].citysIndexs
    if len(cellCitysInxs) == 0:
        return -1
    for i in cellCitysInxs:
        s += float(cityEcomonic[i])
    return s/float(len(cellCitysInxs))

#hexgon shift
shift = {0:2,1:2,2:1,3:1,4:0,5:1,6:1,7:2,8:2}
def display(sample_surface):
    '''
    display the map
    :param sample_surface: sample_surface
    :return: None
    '''
    x1, y1 = 0, 40
    x2, y2 = 0, 20
    x3, y3 = 20, 0
    x4, y4 = 40, 20
    x5, y5 = 40, 40
    x6, y6 = 20, 60

    points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6)]
    for p in range(6):
        points[p] = (points[p][0] + (45 * (shift[0] )), points[p][1])
    for i in range(SizeSom):
        curr_points = copy.deepcopy(points)
        for p in range(6):
            if i<4:
                curr_points[p] = (curr_points[p][0] - (45 * i), curr_points[p][1])
            else:
                if i == 8:
                    curr_points[p] = (curr_points[p][0] - (45 * (12-i+(i%4))), curr_points[p][1])
                else:
                    curr_points[p] = (curr_points[p][0] - (45 * (8-i+(i%4))), curr_points[p][1])

        for j in range(SizeSom - shift[i]*2 + i%2):
            ecomonicAvg = int(getEcomonicAvg(i,j))
            pygame.draw.polygon(sample_surface, chooseColorForHax(ecomonicAvg),
                                curr_points)
            font = pygame.font.SysFont('Arial', 12)
            sample_surface.blit(font.render("("+i.__str__() +","+j.__str__() + ") c:"
                                            + ecomonicAvg.__str__(), True, (0, 0, 0)), curr_points[1])
            pygame.display.update()
            for p in range(6):
                curr_points[p] = (curr_points[p][0] + 45, curr_points[p][1])
        for p in range(6):
            points[p] = (points[p][0] + 25, points[p][1] + 45)

def print_cities_by_cells():
    '''
    for each cell print the cities that were positioned in that cell
    :return: None
    '''
    global somMap,cityName
    map_cityName_to_cell = {}
    for i, row in enumerate(somMap):
        for j, cell in enumerate(row):
            if cell!=None and len(cell.citysIndexs) > 0:
                for cityInx in cell.citysIndexs:
                    if (i,j) not in map_cityName_to_cell.keys():
                        map_cityName_to_cell[(i, j)] = []
                    map_cityName_to_cell[(i,j)].append(cityName[cityInx])
    print(map_cityName_to_cell)

def on_ext():
    '''
    :return: None
    '''
    print_cities_by_cells()

def on_pause():
    '''
    :return: None
    '''
    print_cities_by_cells()

pause = False
def check_event():
    '''
    check for events
    :return: ז
    '''
    global pause
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pause = False
            on_ext()
            sys.exit(0)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                pause = not pause
                on_pause()

if __name__ == '__main__':
    filePath = input('Enter your file path:')
    if filePath == "":
        filePath = "./Elec_24.csv"
    initialize(filePath)
    pygame.init()
    sample_surface = pygame.display.set_mode((500, 500))
    sample_surface.fill((200,100,255))
    count_epoch = 0

    while(True):
        display(sample_surface)
        pygame.display.flip()
        check_event()
        while pause:
            check_event()
        print("start epoch : " + count_epoch.__str__())
        epoch()
        print("done epoch : " + count_epoch.__str__())
        count_epoch +=1
        #time.sleep(0.5)

