import numpy as np
import math
import random
import matplotlib.patches as patches
from matplotlib import pyplot
from scipy import signal
from celluloid import Camera
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import dummy

'''
Simulation Parameters
'''
#Lattice width
W = 14
#Domain Length
L = 105
#VEGF diffusion constant
D = 0.000167
#Cell radius
R = 0.5
#VEGF sensing parameter
xi = 0.1
#VEGF production rate
chi = 0.00000167
#Height weighting parameter
lmbd = 0.167
#Smoothing for boundary conditions
bcParam = 2
#Time of simulation in minutes
simTime = 1800
#Initial VEGF concentration
c0 = 0.5
#Averaging parameter
runs = 10

'''
Create array to hold VEGF (initial value c0)
'''
def createVEGFArray(width, length, c0):
    VEGFArray = np.full((width, length), c0)
    #Impose Dirichlet boundary conditions (c(x,y)_border = 0)
    VEGFArray[0, :] = 0
    VEGFArray[:, 0] = 0
    VEGFArray[-1, :] = 0
    VEGFArray[:, -1] = 0
    #Smoothing for PDE solver
    for i in range (bcParam):
        #Left hand boundary smoothing
        VEGFArray[i:-i, i] = (i / bcParam) * c0
        #Upper boundary smoothing
        VEGFArray[i, i:-i] = (i / bcParam) * c0
        #Right hand boundary smoothing
        VEGFArray[i:-i, -(i + 1)] = (i / bcParam) * c0
        #Lower boundary smoothing
        VEGFArray[-(i + 1), i:-i] = (i / bcParam) * c0
    return (VEGFArray)

'''
Calculate and return diffusion term of the VEGF PDE
'''
def diffusion(VEGFArray, D):
    #2-D Laplacian Kernel
    kernel = [[0.0, 1.0, 0.0],
              [1.0, -4.0, 1.0],
              [0.0, 1.0, 0.0]]
    return D * signal.convolve2d(VEGFArray, kernel, boundary = 'fill', mode = 'same', \
                                fillvalue = 0)

'''
Calculate and return the logistic term of the VEGF PDE
'''
def logistic(VEGFArray, chi, width, length):
    #(1 - c) Array
    logistArray = np.subtract(np.ones((width, length)), VEGFArray)
    #Logistic term in VEGF equation
    return(chi * np.multiply(VEGFArray, logistArray))

'''
Calculate and return the internalisation term of the VEGF PDE
'''
def summation(VEGFArray, cellArray, lmbd, R, width, length):
    #List of cell locations
    indexList = np.nonzero(cellArray)
    #Create empty matrix to store values
    summFinArray = np.zeros((width, length))
    #Iterate over VEGF matrix for individual terms
    for i in range(width):
        for j in range(length):
            #Exponential term in uptake equation
            exFactor = 0 
            for k in range(len(indexList[0])): 
                exFactor += math.exp(-(1 / (2 * R**2)) * ((indexList[0][k] - i)**2 + \
                                                          (indexList[1][k] - j)**2))    
            summFinArray[i, j] = (lmbd / (2 * math.pi * R**2))*VEGFArray[i, j] * exFactor
    #Uptake term of the VEGF equation
    return(summFinArray)

'''
Calculate and return updated VEGF matrix from VEGF PDE
'''
#Update VEGFArray according to governing VEGF PDE    
def updateVEGF(VEGFArray, cellArray, D, chi, width, length, lmbd, R):
    #Updated VEGF array calculated by Taylor expansion
    #c(x, t + delta t) = c(x, t) + delta t * c'(x, t)
    return (VEGFArray + diffusion(VEGFArray, D) + logistic(VEGFArray, chi, width, length) \
            - summation(VEGFArray, cellArray, lmbd, R, width, length))

'''
Insert cell at randomly selected position (x = 0)
'''
#Populate cell matrix with leader/follower cells
def populate(cellArray, width):
    #Generate random y coordinates for potential insertion (without replacement)
    y = np.random.randint(0, width)
    #Attempt to populate array with cells
    if cellArray[y, 0] == 0:
        #Insert leader cell
        cellArray[y, 0] = 1

def chemotaxis(VEGFArray, cellArray, length, width):
    #List of cell locations
    cellList = []
    for i in range(width):
        for j in range(length):
            if cellArray[i, j] == 1:
                #Cell location
                cellList.append([i, j])

    #Shuffle cell list for random order of cell consideration
    random.shuffle(cellList)
    #Move cells chemotactically
    for i in cellList:
        #Randomly select filopodium direction 
        j = np.random.choice(8, 1)
        #VEGF concentration at cell site
        cOld = VEGFArray[i[0], i[1]]

        #Consider above lattice site and ensure cell not at upper boundary
        if j == 0:
            #List of sites spanned by filopodium 
            filList = []
            cellList = []
            #Occupy filopodium list with VEGF concentrations
            for k in range(1, 6):
                #Filopdium detects cell presence and VEGF
                if i[0] - k >= 0:
                    filList.append(VEGFArray[i[0] - k, i[1]])
                    cellList.append(cellArray[i[0] - k, i[1]])
                else:
                    break
            #Cell moves if VEGF favourable
            if len(filList) != 0:
                #Average VEGF detected by filopodium 
                cNew = np.mean(filList)
                #VEGF concentration is favourable
                if ((cNew - cOld) / cOld) >= xi * math.sqrt(c0 / cOld) \
                    and (cellArray[i[0] - 1, i[1]] == 0):
                    cellArray[i[0], i[1]] = 0
                    cellArray[i[0] - 1, i[1]] = 1

        #Consider below-left lattice site and ensure cell not at bottom left of lattice
        elif j == 1:
            #List of sites spanned by filopodium 
            filList = []
            cellList = []
            #Occupy filopodium list with VEGF concentrations
            for k in range(1, 5):
                try:
                    if i[1] - k >= 0:
                        filList.append(VEGFArray[i[0] + k, i[1] - k])
                        cellList.append(cellArray[i[0] + k, i[1] - k])
                    else:
                        break
                except IndexError:
                    break

            if len(filList) != 0:
                #Average VEGF detected by filopodium 
                cNew = np.mean(filList)
                #VEGF concentration is favourable
                if ((cNew - cOld) / cOld) >= xi * math.sqrt(c0 / cOld) \
                    and (cellArray[i[0] + 1, i[1] - 1] == 0):
                    cellArray[i[0], i[1]] = 0
                    cellArray[i[0] + 1, i[1] - 1] = 1

        #Consider below lattice site and ensure cell not at bottom of lattice
        elif j == 2:
            #List of sites spanned by filopodium 
            filList = []
            cellList = []
            #Occupy filopodium list with VEGF concentrations
            for k in range(1, 6):
                try:
                    filList.append(VEGFArray[i[0] + k, i[1]])
                    cellList.append(cellArray[i[0] + k, i[1]])
                except IndexError:
                    break

            if len(filList) != 0:
                #Average VEGF detected by filopodium 
                cNew = np.mean(filList)
                #VEGF concentration is favourable
                if ((cNew - cOld) / cOld) >= xi * math.sqrt(c0 / cOld) \
                    and (cellArray[i[0] + 1, i[1]] == 0):
                    cellArray[i[0], i[1]] = 0
                    cellArray[i[0] + 1, i[1]] = 1

        #Consider below-right lattice site and ensure cell not at bottom right of lattice
        elif j == 3:
            #List of sites spanned by filopodium 
            filList = []
            cellList = []
            #Occupy filopodium list with VEGF concentrations
            for k in range(1, 5):
                try:
                    filList.append(VEGFArray[i[0] + k, i[1] + k])
                    cellList.append(cellArray[i[0] + k, i[1] + k])
                except IndexError:
                    break

            if len(filList) != 0:
                #Average VEGF detected by filopodium 
                cNew = np.mean(filList)
                #VEGF concentration is favourable
                if ((cNew - cOld) / cOld) >= xi * math.sqrt(c0 / cOld) \
                    and (cellArray[i[0] + 1, i[1] + 1] == 0):
                    cellArray[i[0], i[1]] = 0
                    cellArray[i[0] + 1, i[1] + 1] = 1

        #Consider left lattice site and ensure cell not at left of lattice
        elif j == 4:
            #List of sites spanned by filopodium 
            filList = []
            cellList = []
            #Occupy filopodium list with VEGF concentrations
            for k in range(1, 6):
                if i[1] - k >= 0:
                    filList.append(VEGFArray[i[0], i[1] - k])
                    cellList.append(cellArray[i[0], i[1] - k])
                else:
                    break

            if len(filList) != 0: 
                #Average VEGF detected by filopodium 
                cNew = np.mean(filList)
                #VEGF concentration is favourable
                if ((cNew - cOld) / cOld) >= xi * math.sqrt(c0 / cOld) \
                    and (cellArray[i[0], i[1] - 1] == 0):
                    cellArray[i[0], i[1]] = 0
                    cellArray[i[0], i[1] - 1] = 1

        #Consider above left lattice site and ensure cell not at top left of lattice
        elif j == 5:
            #List of sites spanned by filopodium 
            filList = []
            cellList = []
            #Occupy filopodium list with VEGF concentrations
            for k in range(1, 5):
                if (i[0] - k >= 0) and (i[1] - k >= 0):
                    filList.append(VEGFArray[i[0] - k, i[1] - k])
                    cellList.append(cellArray[i[0] - k, i[1] - k])
                else:
                    break

            if len(filList) != 0: 
                #Average VEGF detected by filopodium 
                cNew = np.mean(filList)
                #VEGF concentration is favourable
                if ((cNew - cOld) / cOld) >= xi * math.sqrt(c0 / cOld) \
                    and (cellArray[i[0] - 1, i[1] - 1] == 0):
                    cellArray[i[0], i[1]] = 0
                    cellArray[i[0] - 1, i[1] - 1] = 1

        #Consider right cell and ensure cell not at right of lattice
        elif j == 6:
            #List of sites spanned by filopodium 
            filList = []
            cellList = []
            #Occupy filopodium list with VEGF concentrations
            for k in range(1, 6):
                try:
                    filList.append(VEGFArray[i[0], i[1] + k])
                    cellList.append(cellArray[i[0], i[1] + k])
                except IndexError:
                    break
                
            if len(filList) != 0: 
                #Average VEGF detected by filopodium 
                cNew = np.mean(filList)
                #VEGF concentration is favourable
                if ((cNew - cOld) / cOld) >= xi * math.sqrt(c0 / cOld) \
                    and (cellArray[i[0], i[1] + 1] == 0):
                    cellArray[i[0], i[1]] = 0
                    cellArray[i[0], i[1] + 1] = 1

        #Consider above right lattice site and ensure cell not at top right of lattice
        elif j == 7:
            #List of sites spanned by filopodium 
            filList = []
            cellList = []
            #Occupy filopodium list with VEGF concentrations
            for k in range(1, 5):
                try:
                    if (i[0] - k >= 0):
                        filList.append(VEGFArray[i[0] - k, i[1] + k])
                        cellList.append(cellArray[i[0] - k, i[1] + k])
                    else:
                        break
                except IndexError:
                    break
            if len(filList) != 0: 
                #Average VEGF detected by filopodium 
                cNew = np.mean(filList)
                #VEGF concentration is favourable
                if ((cNew - cOld) / cOld) >= xi * math.sqrt(c0 / cOld) \
                    and (cellArray[i[0] - 1, i[1] + 1] == 0) and i[0] - 1 >= 0:
                    cellArray[i[0], i[1]] = 0
                    cellArray[i[0] - 1, i[1] + 1] = 1


'''
Run simulation and display results in real time
'''
def main(): 
    #Figure and axis for visualisation
    #fig, ax = pyplot.subplots()
    #Get location of colourbar
    #div = make_axes_locatable(ax)
    #Separate axis for colorbar
    #cax = div.append_axes('right', '5%', '5%')
    #Initialise camera for GIF creation
    #camera = Camera(fig)
    #Initialise time variable
    t = 0
    #Initialise cell and VEGF lattices
    cellArray = np.zeros((W, L))
    VEGFArray = createVEGFArray(W, L, c0)
    #Main simulation iteration (minutes)
    for i in range (1, simTime - 1):
        #Solve for VEGF
        VEGFArray = updateVEGF(VEGFArray, cellArray, D, chi, W, L, lmbd, R)
        #Add new cell every 6 minutes
        if t % 6 == 0:
            populate(cellArray, W)
        #Cells move every 12 minutes
        if t % 12 == 0:
            chemotaxis(VEGFArray, cellArray, L, W)
            #Generate image for GIF
            #cax.cla()
            #for j in range(W):
                #for k in range(L):
                    #if cellArray[j, k] == 1:
                        #ax.add_patch(patches.Rectangle((k - 0.5, j - 0.5), 1, 1, \
                                                       #linewidth=1, edgecolor='w', facecolor='w'))
            #im = ax.imshow(VEGFArray, interpolation='none', vmin = 0, vmax = 0.5)
            #ax.set_title('Chemotaxis on Stationary Domain [30h]')
            #fig.colorbar(im, cax=cax)
            #camera.snap()
        #Timestep
        t += 1
    #anim = camera.animate()
    #anim.save('Chemotaxis on Stationary Domain [30h].GIF')
    distList = np.sum(cellArray, axis=0)
    return(distList)
    
def plotter(runs):
    masterList = []
    for i in range(runs):
        distList = main()
        masterList.append(distList)
    plotList = []
    errorList = []
    for i in list(map(list, zip(*masterList))):
        plotList.append(np.mean(i))
        errorList.append(np.std(i) / np.sqrt(np.size(i)))
    pyplot.errorbar(np.arange(0, L), plotList, yerr=errorList, fmt='o')
    pyplot.title('Average Column Occupancy for Chemotaxis on Stationary Domain [30h]')
    pyplot.xlabel('Distance from Neural Tube Exit / 10µm')
    pyplot.ylabel('Average Cell Occupancy at t=30h')
    pyplot.show()
plotter(runs)
