'''Copyright 2022 JoS QUANTUM GmbH

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

from math import sin,cos
import random

# Amplitude Estimation without Phase Estimation. Based on Suzuki et. al.
# Gradient search for the best value for the probability. Gradient search is
# repeated with random starting points.
# Meta parameters: stepSize=0.01 and rounds=10


#
# Cacluate the probability of a single event.
#
def likeli(N,m,h,theta):
    partA=(sin((2*m+1)*theta)**2)**h
    partB=(cos((2*m+1)*theta)**2)**(N-h)
    return partA*partB

#
# Calculate the likelihood function for given vectors N, M and H and a value of theta
# This is the probability of the event that is described by the vectors.
#
def likeliVector(vectorN, vectorM, vectorH, theta):
    product=1.0
    for i in range(len(vectorN)):
        currentN=vectorN[i]
        currentM=vectorM[i]
        currentH=vectorH[i]
        product=product*likeli(currentN,currentM,currentH,theta)
    return product

#
# One gradient search loop. This method should be started with several random points and
# the best result (highest bestProb at the end) should be taken.
#
def gradientOptimizerVector(vectorN, vectorM, vectorH, thetaStart, stepSize):

    # Init the search
    bestTheta=thetaStart
    bestProb=likeliVector(vectorN, vectorM,vectorH,thetaStart)
    continueFlag=True

    # Search as long we find an improvement
    while continueFlag==True:
        thetaPlus=bestTheta+stepSize
        probPlus=likeliVector(vectorN, vectorM,vectorH,thetaPlus)

        thetaMinus=bestTheta-stepSize
        probMinus=likeliVector(vectorN, vectorM,vectorH,thetaMinus)

        bestThetaLocal=thetaPlus
        bestProbLocal=probPlus

        if probMinus>bestProbLocal:
            bestThetaLocal=thetaMinus
            bestProbLocal=probMinus

        if bestProbLocal>bestProb:
            bestTheta=bestThetaLocal
            bestProb=bestProbLocal
        else:
            continueFlag=False
    return bestTheta,bestProb

#
# Run the gradient search loop several times with random start points
#
def loopGradientOptimizerVector(vectorN, vectorM, vectorH, rounds=10, stepSize=0.01):
    theta=random.random()
    bestTheta,bestProb=gradientOptimizerVector(vectorN, vectorM, vectorH, theta, stepSize)

    # Init search with random point
    for i in range(rounds):
        theta=random.random()
        currentTheta,currentProb=gradientOptimizerVector(vectorN, vectorM, vectorH, theta, stepSize)
        if currentProb>bestProb:
            bestTheta=currentTheta
            bestProb=currentProb
    return bestTheta, sin(bestTheta)**2

# Example: Use exakt expectation values instead of counts. Should lead to 0.695.
#vectorN=[30,30,30]
#vectorM=[0,1,2]
#vectorH=[20.849999999999994, 1.0091400000000035, 28.619680776000017]
#bestTheta,bestProb=loopGradientOptimizerVector(vectorN, vectorM, vectorH, 30, 0.001)
#print("best guess theta=",bestTheta)
#print("best guess prob=",sin(bestTheta)**2)
