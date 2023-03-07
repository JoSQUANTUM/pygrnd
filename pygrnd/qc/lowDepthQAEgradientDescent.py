'''Copyright 2023 JoS QUANTUM GmbH

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''


# Functions for gradient descent to fit the angle (and error model) for
# results from a low-depth QAE.

import math
import random

def getSmallerAngle(angle):
    """ For sin(theta/2)**2 return the smaller angle of the two possible
        values that lead to the same probability.
    """
    angle1=angle
    angle2=2*math.pi-angle
    return min(angle1,angle2)

#
# Functions without the exponential error model.
#

def expectedProbabilityNoErrorModel(theta, ell):
    ''' Return the expected probability for a good state for angle theta
        and ell Grover operators after the initialization. No error model.
    '''
    return math.sin((2*ell+1)*theta/2)**2

def errorAllPointsNoErrorModel(theta, vectorN, vectorM, vectorR):
    ''' Return the sum of the squared deviations of the expected
        counts of good results and the measured results for the
        angle theta. vectorN contains the total counts, vectorM
        contains the number of Grover operators after the initialization
        and vectorR contains the counts of good states from the measurements.
        No error model.
    '''
    error=0.0
    for i in range(len(vectorN)):
        probMeasured=vectorR[i]/vectorN[i]
        probExpected=expectedProbabilityNoErrorModel(theta, vectorM[i])
        error=error+(probMeasured-probExpected)**2
    return error

def gradientOptimizerVectorNoErrorModel(vectorN, vectorM, vectorR, thetaStart, stepSize, learningRate):
    ''' For given lists vectorN, vectorM and vectorR perform a gradient
        descent search for the angle theta that minimizes errorAllPoints.
        The method returns the angle and the corresponding error from the
        deviation of the parameter fitting. Other parameters are the step
        size for the gradient approximation, the learning rate and the start value for theta.
    '''

    # Initialize the search.
    bestTheta=thetaStart
    bestError=errorAllPointsNoErrorModel(thetaStart, vectorN, vectorM, vectorR)
    continueFlag=True

    # Search as long we find an improvement.
    while continueFlag==True:
        bestThetaPrime=bestTheta+stepSize
        error1=errorAllPointsNoErrorModel(bestThetaPrime, vectorN, vectorM, vectorR)

        gradientUnnormed=[(error1-bestError)/stepSize]
        gradientNorm=sum(abs(x)**2 for x in gradientUnnormed)
        gradientNormed=[x/math.sqrt(gradientNorm) for x in gradientUnnormed]

        thetaNew=bestTheta-learningRate*gradientNormed[0]

        errorNew=errorAllPointsNoErrorModel(thetaNew, vectorN, vectorM, vectorR)

        if errorNew+0.000001<bestError:
            bestError=errorNew
            bestTheta=thetaNew
        else:
            continueFlag=False

    return bestTheta, bestError

def loopGradientOptimizerVectorNoErrorModel(vectorN, vectorM, vectorR, rounds=10, stepSize=0.0001, learningRate=0.0001):
    ''' Run the gradient search for given vectorN, vectorM and vectorR. The search
        starts with random angles and has the specified number of rounds and step size.
        Returns the angle along with the corresponding probability estimation for the
        best result that was found.
    '''

    # Initialize search with random point.
    theta=2*math.pi*random.random()
    bestTheta,bestError=gradientOptimizerVectorNoErrorModel(vectorN, vectorM, vectorR, theta, stepSize, learningRate)

    for i in range(rounds):
        theta=2*math.pi*random.random()
        currentTheta,currentError=gradientOptimizerVectorNoErrorModel(vectorN, vectorM, vectorR, theta, stepSize, learningRate)
        if currentError<bestError:
            bestTheta=currentTheta
            bestError=currentError
    return bestTheta, math.sin(bestTheta/2)**2

#
# Example for the gradient search of theta without error model.
#
# vectorN=[30, 30, 30, 30, 30, 30, 30, 30, 30]
# vectorM=[ 0,  1,  2,  3,  4,  5,  6,  7,  8]
# vectorR=[10, 29,  3, 21, 23,  0, 28, 10,  7]
#
# loopGradientOptimizerVectorNoErrorModel(vectorN, vectorM, vectorR, rounds=100, stepSize=0.0001)


#
# Functions for the exponential error model.
#

def expectedProbabilityErrorModel(theta, ell, a, f):
    ''' Return the expected probability for a good state for angle theta
        and ell Grover operators after the initialization. The error model
        has the parameters a and f.
    '''
    return  math.exp(-a*ell)*math.sin((2*ell+1)*theta/2)**2 + (1-math.exp(-a*ell))*f

def errorAllPointsErrorModel(theta, a, f, vectorN, vectorM, vectorR):
    ''' Return the sum of the squared deviations of the expected
        counts of good results and the measured results for the
        angle theta and parameters a and f of the error model. vectorN contains
        the total counts, vectorM contains the number of Grover operators
        after the initialization and vectorR contains the counts of good states
        from the measurements.
    '''
    error=0.0
    for i in range(len(vectorN)):
        probMeasured=vectorR[i]/vectorN[i]
        probExpected=expectedProbabilityErrorModel(theta, vectorM[i], a, f)
        error=error+(probMeasured-probExpected)**2
    return error

def gradientOptimizerVectorErrorModel(vectorN, vectorM, vectorR, thetaStart, aStart, fStart, stepSize, learningRate):
    ''' For given lists vectorN, vectorM and vectorR perform a gradient
        descent search for the angle theta and parameters a and f of the error model
        that minimizes errorAllPoints. The method returns the angle and the parameters of the
        error model and the corresponding error from the deviation of the parameter fitting.
        Other parameters are the step size for the gradient approximation, the learning rate and
        the start value for theta and the parameters of the error model.
    '''

    # Initialize the search.
    bestTheta=thetaStart
    bestA=aStart
    bestF=fStart
    bestError=errorAllPointsErrorModel(thetaStart, aStart, fStart, vectorN, vectorM, vectorR)
    continueFlag=True

    # Search as long we find an improvement.
    while continueFlag==True:
        bestThetaPrime=bestTheta+stepSize
        error1=errorAllPointsErrorModel(bestThetaPrime, bestA, bestF, vectorN, vectorM, vectorR)

        bestAPrime=bestA+stepSize
        error2=errorAllPointsErrorModel(bestTheta, bestAPrime, bestF, vectorN, vectorM, vectorR)

        bestFPrime=bestF+stepSize
        error3=errorAllPointsErrorModel(bestTheta, bestA, bestFPrime, vectorN, vectorM, vectorR)

        gradientUnnormed=[(error1-bestError)/stepSize,(error2-bestError)/stepSize,(error3-bestError)/stepSize]
        gradientNorm=sum(abs(x)**2 for x in gradientUnnormed)
        gradientNormed=[x/math.sqrt(gradientNorm) for x in gradientUnnormed]

        thetaNew=bestTheta-learningRate*gradientNormed[0]
        aNew=bestA-learningRate*gradientNormed[1]
        fNew=bestF-learningRate*gradientNormed[2]

        errorNew=errorAllPointsErrorModel(thetaNew, aNew, fNew, vectorN, vectorM, vectorR)

        if errorNew+0.000001<bestError:
            bestError=errorNew
            bestTheta=thetaNew
            bestA=aNew
            bestF=fNew
        else:
            continueFlag=False

    return bestTheta, bestA, bestF, bestError

def loopGradientOptimizerVectorErrorModel(vectorN, vectorM, vectorR, rounds=10, stepSize=0.001, learningRate=0.001):
    ''' Run the gradient search for given vectorN, vectorM and vectorR. The search
        starts with random angles and parameters for the error model and it has the specified number
        of rounds and step size. Returns the angle and the parameters of the error model
        along with the corresponding probability estimation for the best result that was found.
    '''

    # Initialize search with random point.
    theta=random.random()
    a=random.random()
    f=random.random()
    bestTheta, bestA, bestF,bestError=gradientOptimizerVectorErrorModel( vectorN, vectorM, vectorR, theta, a, f, stepSize, learningRate)

    for i in range(rounds):
        theta=2*math.pi*random.random()
        a=random.random()
        f=random.random()
        currentTheta, currentA, currentF,currentError=gradientOptimizerVectorErrorModel( vectorN, vectorM, vectorR, theta, a, f, stepSize, learningRate)
        if currentError<bestError:
            bestTheta=currentTheta
            bestA=currentA
            bestF=currentF
            bestError=currentError
    return bestTheta, bestA, bestF, math.sin(bestTheta/2)**2

#
# Example for the gradient search of theta with error model.
#

# vectorN=[30, 30, 30, 30, 30, 30, 30, 30, 30]
# vectorM=[ 0,  1,  2,  3,  4,  5,  6,  7,  8]
# R1=[10, 29, 3, 21, 23, 0, 28, 10, 7]
# R2=[15, 20, 13, 16, 12, 23, 6, 27, 6]
# R3=[16, 10, 29, 1, 30, 2, 25, 10, 24]
# R4=[21, 3, 30, 3, 22, 14, 1, 30, 5]
# R5=[19, 0, 27, 10, 11, 29, 0, 19, 20]
# Rs=[R1,R2,R3,R4,R5]
#
# for i in range(len(Rs)):
#     resTheta,resA,resF,resProb=loopGradientOptimizerVectorErrorModel(vectorN, vectorM, Rs[i], rounds=10, stepSize=0.001, learningRate=0.001)
#     print("theta/a/f/prob=",resTheta,resA,resF,resProb)
