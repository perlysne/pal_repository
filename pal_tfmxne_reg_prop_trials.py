#
# This function multivariate regresses the timecourse output of tf_mixed_norm
# onto the evoked response at the sensors. Since tf_mixed_norm is limited to
# average evoked responses, this is an attempt to create a pseudo inverse
# operator to order to generate single trial output (which is needed for
# spectral estimation).
#
# The system to be solved is: X=MB, where B is given by: B=(inv(M'M))M'X
#
# This system is proposed as the inverse of M=GX, the forward model of Gramfort
# et al (2013).
#
# Inputs:
#     evoked_data: the sensor data array as read from the evoked response file
#                  (expected size: NSensors x NSamples)
#                  (in my pipeline the evoked response is generated via the 
#                   pre-compiled linux binaries)
#     epochs_data: the sensor level epochs data
#                  (expected size: NTrials x NSensors x NSamples)
#                  (in my pipeline the epochs are extracted from the continuous
#                   data via the python constructor mne.Epochs)
#     tfmxne_data: the source space timecourses output from tf_mixed_norm for
#                  the average evoked response       
#                  (expected size: NSources x NSamples)
#
# Outputs:
#     B:           the multivariate regression weights 
#                  (expected size: NSensors+1(intercept) x NSources
#     trials_data: the individual trial data for the TF-MxNE localizations
#                  (expected size: NTrials x NSources x NSamples)
#
# Currently the multivariate regression approach implemented here works
# for neuromag data using magnetometers only, but not neuromag gradiometers
# or combined data. Does not work for CTF data, which is gradiometers only.
#
# Per A. Lysne
# University of New Mexico
# lysne@unm.edu
# 10/15/2014
#

import sys
import ipdb
import math
import numpy
import matplotlib

from pal_wilks_lambda import pal_wilks_lambda

def pal_tfmxne_reg_prop_trials(evoked_data, epochs_data, tfmxne_data):

    DebugFlag=1

    print "pal_tfmxne_reg_prop_trials: entering..."
    print "    evoked_data.shape: ", evoked_data.shape
    print "    epochs_data.shape: ", epochs_data.shape
    print "    tfmxne_data.shape: ", tfmxne_data.shape

    #
    # M is the average evoked response in sensor space. Note that this is the
    # evoked data with no preprocessing such as whitening, etc. For this reason
    # 'M' here is not the same as the 'M' inside tf_mixed_norm().
    #
    # Obvervations are rows.
    #
    M = evoked_data.T
    M = numpy.append(M,numpy.ones((M.shape[0],1)),axis=1) # Add Intercept Term
    print "M.shape:", M.shape

    #
    # X are the timecourses of the TF-MxNE localizations based on the average
    # evoked response. This is the 'X' that is output from tf_mixed_norm..
    #
    X = tfmxne_data.T
    print "X.shape:", X.shape

    #
    # Perform multivariate regression of the TF-MxNE timecourses on the
    # evoked average sensor data. We are solving the system M=XB, and the
    # solution is given by B = (inv(M'M))M'X.
    #
    # B are the linear weights which will be used to transform the sensor
    # space trials into the sparse TF-MxNE source space timecourses.
    #
    print "pal_tfmxne_reg_prop_trials: regressing TF-MxNE timecourses on evoked average sensor timecourses..."
    Tmp = numpy.dot(M.T,M)
    print "Tmp.shape:", Tmp.shape

    Tmp = numpy.linalg.inv(Tmp)
    print "Tmp.shape:", Tmp.shape

    Tmp = numpy.dot(Tmp,M.T)
    print "Tmp.shape:", Tmp.shape

    B = numpy.dot(Tmp,X)
    print "B.shape:", B.shape

    NSources = X.shape[1]
    NTrials  = epochs_data.shape[0]
    NSamples = epochs_data.shape[2]

    #
    # Sanity check our multivariate regression by propagating the
    # evoked average itself (the evoked average sensor level
    # timecourses are what we just used as the predictors in the
    # regression).
    #
    print "pal_tfmxne_reg_prop_trials: testing regression result on the evoked average..."
    MT = evoked_data.T

    MT = numpy.append(MT,numpy.ones((MT.shape[0],1)),axis=1) # Add Intercept Term
    print "MT.shape: ", MT.shape
    X_est = numpy.dot(MT,B)
    print "X_est.shape: ", X_est.shape

    if DebugFlag:
        matplotlib.pyplot.figure
        for s in range(0,NSources):
            matplotlib.pyplot.plot(X    [:,s],'r',label='X'+str(s))        # TF-MxNE output timecourses

        for s in range(0,NSources):
            matplotlib.pyplot.plot(X_est[:,s],'b',label='X'+str(s)+'_est') # estimated source timecourses

        matplotlib.pyplot.xlabel("Samples")
        matplotlib.pyplot.ylabel("Source Amplitude")
        matplotlib.pyplot.title("TF-MxNE Timecourses (red) and Regression Estimates (blue)")
        matplotlib.pyplot.legend()

    #
    # Propagate the sensor-level epochs through our solution to get
    # the source-level individual trials.
    #
    print "pal_tfmxne_reg_prop_trials: propagating trials..."

    trials_data = numpy.zeros((NTrials, NSources, NSamples), dtype="float")
    #print "trials_data.shape:", trials_data.shape

    for t in range(0,NTrials):
        print "propagating trial: ", t
        MT = epochs_data[t,:,:]
        MT = MT.T
        MT = numpy.append(MT,numpy.ones((MT.shape[0],1)),axis=1) # Add Intercept Term
        print "MT.shape:", MT.shape
        print "B.shape: ", B.shape

        XT = numpy.dot(MT,B)
        print "XT.shape:", XT.shape

        trials_data[t,:,:] = XT.T

        if (0):
            for s in range(0,NSources):
                print "    subtracting temporal mean from trial:", t, "source:", s
                trials_data[t,s,:] = numpy.subtract(trials_data[t,s,:],trials_data[t,s,:].mean(axis=0))

    #
    # Plot the individual trials and the mean for each source.
    #
    if DebugFlag:
        matplotlib.pyplot.figure()

        for s in range(0,NSources):

            SubPlotDim1 = math.ceil(math.sqrt(NSources))
            SubPlotDim2 = math.ceil(NSources/SubPlotDim1)
            ax=matplotlib.pyplot.subplot(SubPlotDim2,SubPlotDim1,s)

            for t in range (0,NTrials):
                matplotlib.pyplot.plot(trials_data[t,s,:],color='b')

            matplotlib.pyplot.plot(trials_data[:,s,:].mean(axis=0),color='r',linewidth=2)

            matplotlib.pyplot.xlabel('Samples')
            matplotlib.pyplot.ylabel('Source Amplitude')
            matplotlib.pyplot.title('Source '+str(s))

    matplotlib.pyplot.suptitle('Source Average Timecourses and Individual Trials')

    print "pal_tfmxne_reg_prop_trials: leaving..."
    print "    B.shape:          ", B.shape
    print "    trials_data.shape:", trials_data.shape

    #
    # Return the multivariate regression weights and the source-level
    # individual trial timecourses.
    #
    return(B, trials_data)
