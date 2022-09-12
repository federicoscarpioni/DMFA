'''
Scritp to stream sampled data from digital oscilloscope Picoscope 4000a series. 
Three channels are used here.
Adapted from the example by PicoTechnology
Authors: Federico Scarpioni and Nicolò Pianta
'''

# Folder path for the saving of the sampled signals
saving_path = 'C:/data_aquisition'

import ctypes
import numpy as np
from picosdk.ps4000a import ps4000a as ps
from picosdk.functions import adc2mV, assert_pico_ok
import time

# Show the starting time
t_initial = time.process_time()
print('Measurement started on ' + time.ctime())

# Create chandle and status ready for use
chandle = ctypes.c_int16()
status = {}

# Open PicoScope 4000a Series device
# Returns handle to chandle for use in future API functions
status["openunit"] = ps.ps4000aOpenUnit(ctypes.byref(chandle), None)
try:
    assert_pico_ok(status["openunit"])
except:

    powerStatus = status["openunit"]

    if powerStatus == 286:
        status["changePowerSource"] = ps.ps4000aChangePowerSource(chandle, powerStatus)
    else:
        raise

    assert_pico_ok(status["changePowerSource"])
enabled = 1
disabled = 0
analogue_offset = 0.0

# Channel voltage ranges vector as a refernce to select the number below (in mV)
# [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
#  0   1   2   3    4    5    6     7     8      9     10     11      12      13

# Set up channel A
channelA_range = 8
status["setChA"] = ps.ps4000aSetChannel(chandle,
                                        ps.PS4000A_CHANNEL['PS4000A_CHANNEL_A'],
                                        enabled,
                                        ps.PS4000A_COUPLING['PS4000A_DC'],
                                        channelA_range,
                                        analogue_offset)
assert_pico_ok(status["setChA"])

# Set up channel B
channelB_range = 4
status["setChB"] = ps.ps4000aSetChannel(chandle,
                                        ps.PS4000A_CHANNEL['PS4000A_CHANNEL_B'],
                                        enabled,
                                        ps.PS4000A_COUPLING['PS4000A_DC'],
                                        channelB_range,
                                        analogue_offset)
assert_pico_ok(status["setChB"])

# Set up channel C
channelC_range = 9
status["setChC"] = ps.ps4000aSetChannel(chandle,
                                        ps.PS4000A_CHANNEL['PS4000A_CHANNEL_C'],
                                        enabled,
  
                                        
  ps.PS4000A_COUPLING['PS4000A_DC'],
                                        channelC_range,
                                        analogue_offset)
assert_pico_ok(status["setChC"])

# Size of capture
sizeOfOneBuffer = 3000
numBuffersToCapture = 1
totalSamples = sizeOfOneBuffer * numBuffersToCapture

# Create buffers ready for assigning pointers for data collection
bufferAMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)
bufferBMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)
bufferCMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)

memory_segment = 0

# Set data buffer location for data collection from channel A
status["setDataBuffersA"] = ps.ps4000aSetDataBuffers(chandle,
                                                     ps.PS4000A_CHANNEL['PS4000A_CHANNEL_A'],
                                                     bufferAMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                     None,
                                                     sizeOfOneBuffer,
                                                     memory_segment,
                                                     ps.PS4000A_RATIO_MODE['PS4000A_RATIO_MODE_NONE'])
assert_pico_ok(status["setDataBuffersA"])

# Set data buffer location for data collection from channel B
status["setDataBuffersB"] = ps.ps4000aSetDataBuffers(chandle,
                                                     ps.PS4000A_CHANNEL['PS4000A_CHANNEL_B'],
                                                     bufferBMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                     None,
                                                     sizeOfOneBuffer,
                                                     memory_segment,
                                                     ps.PS4000A_RATIO_MODE['PS4000A_RATIO_MODE_NONE'])
assert_pico_ok(status["setDataBuffersB"])

# Set data buffer location for data collection from channel C

status["setDataBuffersC"] = ps.ps4000aSetDataBuffers(chandle,
                                                     ps.PS4000A_CHANNEL['PS4000A_CHANNEL_C'],
                                                     bufferCMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                     None,
                                                     sizeOfOneBuffer,
                                                     memory_segment,
                                                     ps.PS4000A_RATIO_MODE['PS4000A_RATIO_MODE_NONE'])
assert_pico_ok(status["setDataBuffersC"])

# Begin streaming mode:
sampleInterval = ctypes.c_int32(100000)
sampleUnits = ps.PS4000A_TIME_UNITS['PS4000A_US']
# We are not triggering:
maxPreTriggerSamples = 0
autoStopOn = 1
# No downsampling:
downsampleRatio = 1
status["runStreaming"] = ps.ps4000aRunStreaming(chandle,
                                                ctypes.byref(sampleInterval),
                                                sampleUnits,
                                                maxPreTriggerSamples,
                                                totalSamples,
                                                autoStopOn,
                                                downsampleRatio,
                                                ps.PS4000A_RATIO_MODE['PS4000A_RATIO_MODE_NONE'],
                                                sizeOfOneBuffer)
assert_pico_ok(status["runStreaming"])

actualSampleInterval = sampleInterval.value
actualSampleIntervalNs = actualSampleInterval * 1000

print("Capturing at sample interval %s ns" % actualSampleIntervalNs)

# We need a big buffer, not registered with the driver, to keep our complete capture in.
bufferCompleteA = np.zeros(shape=totalSamples, dtype=np.int16)
bufferCompleteB = np.zeros(shape=totalSamples, dtype=np.int16)
bufferCompleteC = np.zeros(shape=totalSamples, dtype=np.int16)
nextSample = 0
autoStopOuter = False
wasCalledBack = False


def streaming_callback(handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param):
    global nextSample, autoStopOuter, wasCalledBack
    wasCalledBack = True
    destEnd = nextSample + noOfSamples
    sourceEnd = startIndex + noOfSamples
    bufferCompleteA[nextSample:destEnd] = bufferAMax[startIndex:sourceEnd]
    bufferCompleteB[nextSample:destEnd] = bufferBMax[startIndex:sourceEnd]
    bufferCompleteC[nextSample:destEnd] = bufferCMax[startIndex:sourceEnd]
    nextSample += noOfSamples
    if autoStop:
        autoStopOuter = True


# Convert the python function into a C function pointer.
cFuncPtr = ps.StreamingReadyType(streaming_callback)

# Fetch data from the driver in a loop, copying it out of the registered buffers and into our complete one.
while nextSample < totalSamples and not autoStopOuter:
    wasCalledBack = False
    status["getStreamingLastestValues"] = ps.ps4000aGetStreamingLatestValues(chandle, cFuncPtr, None)
    if not wasCalledBack:
        # If we weren't called back by the driver, this means no data is ready. Sleep for a short while before trying
        # again.
        time.sleep(0.01)

print("Done grabbing values on "+time.ctime()+'. This took '+str(time.process_time() - t_initial)+' s')
t_adv2mV = time.time()

# Find maximum ADC count value
maxADC = ctypes.c_int16()
status["maximumValue"] = ps.ps4000aMaximumValue(chandle, ctypes.byref(maxADC))
assert_pico_ok(status["maximumValue"])

# Convert ADC counts data to mV
t_translate = time.time()
    
channelInputRanges = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]

# Convert ADC values to voltages
adc2mVChAMax = np.multiply(- bufferCompleteA, (channelInputRanges[channelA_range]/maxADC.value), dtype = 'float32')
adc2mVChBMax = np.multiply(- bufferCompleteB, (channelInputRanges[channelB_range]/maxADC.value), dtype = 'float32')
adc2mVChCMax = np.multiply(- bufferCompleteC, (channelInputRanges[channelC_range]/maxADC.value), dtype = 'float32')


print('Elapsed time for translating the buffer to mV: ' + str(time.time() - t_translate) + ' s')

# Stop the scope
# handle = chandle
status["stop"] = ps.ps4000aStop(chandle)
assert_pico_ok(status["stop"])

# Disconnect the scope
# handle = chandle
status["close"] = ps.ps4000aCloseUnit(chandle)
assert_pico_ok(status["close"])

# Display status returns
print(status)

print('Done converting from adv to mV on '+str(time.ctime())+'. This took '+str(time.time()-t_adv2mV)+' s')
print('The whole process took '+str(time.time()-t_initial)+' s')


#%% Save sampled signals in npy files in parallel
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=(3))
time_save = time.time()
a = executor.submit(np.save, saving_path + '/CHA.npy', adc2mVChAMax)
b = executor.submit(np.save, saving_path + '/CHB.npy', adc2mVChBMax)
c = executor.submit(np.save, saving_path + '/CHC.npy', adc2mVChCMax)
a.result()
b.result()
c.result()
print('Saving the data as npy arrays took '+str(time.time()-time_save)+' s')