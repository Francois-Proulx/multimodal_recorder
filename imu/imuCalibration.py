import os
import sys
import time
import smbus
import numpy as np

from imusensor.MPU9250 import MPU9250

address = 0x68
bus = smbus.SMBus(1)
imu = MPU9250.MPU9250(bus, address)

imu.begin()
print ("Accel calibration starting")
imu.caliberateAccelerometer()
print ("Accel calibration Finisehd")
print (imu.AccelBias)
print (imu.Accels)

print ("Mag calibration starting")
time.sleep(2)
# imu.caliberateMagApprox()
imu.caliberateMagPrecise()
print ("Mag calibration Finished")
print (imu.MagBias)
print (imu.Magtransform)
print (imu.Mags)

calib_file = os.path.join(os.path.dirname(__file__), "calibMPU9250.json")
imu.saveCalibDataToFile(calib_file)


# imu.begin()
# imu.caliberateAccelerometer()
# print ("Acceleration calib successful")
# imu.caliberateMagPrecise()
# print ("Mag calib successful")

# accelscale = imu.Accels
# accelBias = imu.AccelBias
# gyroBias = imu.GyroBias
# mags = imu.Mags 
# magBias = imu.MagBias

# imu.saveCalibDataToFile("/home/pi/Documents/frankp/imu/calibMPU9250.json")
# print ("calib data saved")

# imu.loadCalibDataFromFile("/home/pi/Documents/frankp/imu/calibMPU9250.json")
# if np.array_equal(accelscale, imu.Accels) & np.array_equal(accelBias, imu.AccelBias) & \
# 	np.array_equal(mags, imu.Mags) & np.array_equal(magBias, imu.MagBias) & \
# 	np.array_equal(gyroBias, imu.GyroBias):
# 	print ("calib loaded properly")