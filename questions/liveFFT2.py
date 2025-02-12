#!/usr/bin/env python
#coding: utf8

"""
Code originally by Brian R. Pauw and David Mannicke.
Modified by James Tompkin for Brown CSCI1430.

Initial Python coding and refactoring:
	Brian R. Pauw
With input from:
	Samuel Tardif

Windows compatibility resolution: 
	David Mannicke
	Chris Garvey

Windows compiled version:
	Joachim Kohlbrecher
"""

"""
Overview
========
This program uses OpenCV to capture images from the camera, Fourier transform them and show the Fourier transformed image alongside the original on the screen.

Output image visualization:
    Top left: input image
    Bottom left: amplitude image of Fourier decomposition
    Bottom right: phase image of Fourier decomposition
    Top right: reconstruction of image from Fourier domain

$ ./liveFFT2.py

Required: A Python 3.x installation (tested on 3.9.19),
with: 
    - OpenCV (for camera reading)
    - numpy, matplotlib, scipy, argparse
"""

__author__ = "Brian R. Pauw, David Mannicke; modified for Brown CSCI 1430 by James Tompkin"
__contact__ = "brian@stack.nl; james_tompkin@brown.edu"
__license__ = "GPLv3+"
__date__ = "2014/01/25; modifications 2017 onwards"
__status__ = "v2.1"

import os
import cv2 # opencv-based functions
import time
import math
import numpy as np
from scipy import ndimage
from skimage import io
from skimage import img_as_float32, img_as_ubyte
from skimage.color import rgb2gray


class live_FFT2():

    wn = "Fourier decomposition demo"
    use_camera = True

    im = None
    imMerlinTheCat = None
    imRaraMama = None
    
    pause = False
    quit = False
    drawText = True
    blankInput = False

    hw2part = 0
    phaseOffset = 0
    # Variables for animating basis reconstruction
    animateMagnitude = True
    animateOrientation = False
    frequencyCutoffDist = 1
    frequencyCutoffDirection = 0.3
    # Variables for animated basis demo
    magnitude = 1
    orientation = 0

    def __init__(self, **kwargs):

        self.writeControls()

        # Set the name of the output window
        cv2.namedWindow(self.wn, cv2.WINDOW_AUTOSIZE)

        # Load two images
        im1f = 'images/GiraffeCrop.jpg'
        im2f = 'images/MerlinCrop.jpg'
        im1f = im1f if os.path.isfile(im1f) else 'questions/' + im1f
        im2f = im2f if os.path.isfile(im2f) else 'questions/' + im2f
        self.imRaraMama = rgb2gray(img_as_float32(io.imread(im1f)))
        self.imMerlinTheCat = rgb2gray(img_as_float32(io.imread(im2f)))

        # Initialize camera
        # The argument is the device id. 
        # If you have more than one camera, you can access them 
        # by passing a different id, e.g., cv2.VideoCapture(1)
        self.vc = cv2.VideoCapture(0)
        if not self.vc.isOpened():
            print( "No camera found or error opening camera; using a static image instead." )
            self.use_camera = False
            self.im = self.imRaraMama
        else:
            # We found and opened a camera!
            # Requested camera size. This will be cropped to square later on at 240 x 240
            ret = self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            ret = self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

            # Some cameras will return 'None' on read until they are initialized, 
            # so sit and wait for a valid image.
            while self.im is None:
                rval, self.im = self.vc.read()

        # Initial values in case something screws up...
        amplitude = np.zeros( self.im.shape )
        phase = np.zeros( self.im.shape )
        imr = np.zeros( self.im.shape )

        # Main loop
        while True:
            #a = time.perf_counter()
            
            if not self.pause:
                im = self.readAndCropImage()
                amplitude, phase = self.fourierTransform(im)
                amplitude, phase = self.studentFuncs(amplitude, phase)
                imr = self.invFourierTransform(amplitude, phase)
            
            self.displayImages(im, amplitude, phase, imr)
            
            # Update window, pause for 1ms, check keyboard
            key = cv2.waitKey(1)
            self.readKeyboard(key)
            #print('framerate = {} fps \r'.format(1. / (time.perf_counter() - a)))

            if self.quit:
                break
    
        if self.use_camera:
            # Stop camera
            self.vc.release()

    def writeControls(self):

        print("""
            CSCI 1430 Fourier Transform Demo 
            --------------------------------

            0,1,2,3,4,5 = Homework parts --- the buttons won't work until you uncomment each
            p = Toggle pause of camera read and processing
            t = Toggle text
              
            o = Part 1: Animate frequency orientation (default false)
            m = Part 1: Animate frequency magnitude (default true)

            To add your own controls, edit `readKeyboard()`

              """)

        return

    def readKeyboard(self, key):

        for k in range(0, 6):
            if key == ord(str(k)):
                self.hw2part = k
                if self.hw2part == 1:
                    self.blankInput = True
                else:
                    self.blankInput = False
    
        if key == ord('t'):
            self.drawText = not self.drawText
        elif key == ord('p'):
            self.pause = not self.pause
        elif key == ord('o'):
            self.animateOrientation = not self.animateOrientation
        elif key == ord('m'):
            self.animateMagnitude = not self.animateMagnitude
        elif key == 27:
            self.quit = True

        return

    def readAndCropImage(self):

        if self.use_camera:
            # Read image.
            rval, self.im = self.vc.read()

            # Convert to grayscale float32
            tmp = rgb2gray(img_as_float32(self.im))

            # Some students' cameras did not return a 320 x 240 image. 
            # If you have an error about image sizes not matching for Part 3, try uncommenting this line.
            if tmp.shape != (240,320):
                tmp = cv2.resize(tmp, (320,240), interpolation=cv2.INTER_AREA)

            # Crop to square
            # 1. It's not necessary as a square; just easier for didactic reasons
            # 2. Some cameras across the class are returning different image sizes
            #    so let's ensure to crop
            if tmp.shape[1] > tmp.shape[0]:
                cropx = int((tmp.shape[1]-tmp.shape[0])/2)
                cropy = 0
            elif tmp.shape[0] > tmp.shape[1]:
                cropx = 0
                cropy = int((tmp.shape[0]-tmp.shape[1])/2)

            self.im = tmp[cropy:tmp.shape[0]-cropy, cropx:tmp.shape[1]-cropx]

        return self.im

    def fourierTransform(self, im):

        # Let's start by peforming the 2D fast Fourier decomposition operation
        imFFT = np.fft.fft2(im)
        
        # Then creating our amplitude and phase images
        amplitude = np.sqrt(np.power(imFFT.real, 2) + np.power(imFFT.imag, 2))
        phase = np.arctan2(imFFT.imag, imFFT.real)
        
        # NOTE: We will reconstruct the image from this decomposition later on (See Part 5)

        return amplitude, phase

    def invFourierTransform(self, amplitude, phase):
        
        # We need to build a new real+imaginary number from the amplitude / phase
        # This is going from polar coordinates to Cartesian coordinates in the complex number space
        recReal = np.cos( phase ) * amplitude
        recImag = np.sin( phase ) * amplitude
        rec = recReal + 1j*recImag
        
        # Now inverse Fourier transform
        return np.fft.ifft2( rec )

    def displayImages(self, im, amplitude, phase, imr):

        # For visualization, increase brightness of amplitude
        # Catch zero values to prevent any log(0) errors on the next line
        amplitude[amplitude == 0] = np.finfo(float).eps
        # Just for visualization, make it possible to see it
        amplitude = np.log(np.fft.fftshift(amplitude)) / 10
        # Move 
        phase = np.fft.fftshift(phase)

        inputImg = np.zeros(self.im.shape) if self.blankInput else im

        # Take just the real component of the reconstructed image: imr.real
        outputTop = np.concatenate((inputImg, imr.real),axis = 1)
        outputBottom = np.concatenate((amplitude, phase),axis = 1)
        output = np.clip(np.concatenate((outputTop,outputBottom),axis = 0),0,1)

        output = (output*255).astype(np.uint8)
        output_cv_mat = cv2.UMat(output)

        # Write image labels
        if self.drawText:
            position = (0, 0)  # Coordinates of the text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = 255  # White color
            thickness = 2
            if self.blankInput:
                cv2.putText( output_cv_mat, "! Unused !",   (0,25), font, font_scale, font_color, thickness)
            else:
                cv2.putText( output_cv_mat, "Input",        (0,25), font, font_scale, font_color, thickness)
            cv2.putText( output_cv_mat, "Inv. FT",          (int(output.shape[0]/2),25), font, font_scale, font_color, thickness)
            cv2.putText( output_cv_mat, "Fourier Ampli",    (0,25+int(output.shape[1]/2)), font, font_scale, font_color, thickness)
            cv2.putText( output_cv_mat, "Fourier Phase",    (int(output.shape[0]/2),25+int(output.shape[1]/2)), font, font_scale, font_color, thickness)

        # NOTE: One student's software crashed at this line above without casting to uint8,
        # but this operation via img_as_ubyte is _slow_. Add this back in if you code crashes.
        #cv2.imshow(self.wn, img_as_ubyte(output))
        cv2.imshow(self.wn, output_cv_mat)

        return
    
    '''
    Students: Concentrate here.
    '''
    def studentFuncs(self, amplitude, phase):
        
        # Part 1: Scanning the basis and looking at the reconstructed image for each frequency independently
        # ==================================================================================================
        if self.hw2part == 1:
            '''
            # To see the effect, uncomment this block, read through the comments and code, and then execute the program.

            # Lets try to set one basis sine wave to have any amplitude - just like the 'white dot on black background' images in lecture
            # First, let's zero out the amplitude and phase everywhere
            # We'll use some temp variables.
            a = np.zeros( self.im.shape )
            p = np.zeros( self.im.shape )

            # Let's set some controls to animate how it looks as we move through the frequency space
            if self.animateOrientation:
                self.orientation += math.pi / 60.0 # angular speed control
                if self.orientation > math.pi * 2:
                    self.orientation = 0
                
            if self.animateMagnitude:
                self.magnitude += 0.3
                if self.magnitude >= 50: # could go to width/2 for v. high frequencies
                    self.magnitude = 0

            # Make a little vector from the center of the image
            cx = math.floor(self.im.shape[1]/2)
            cy = math.floor(self.im.shape[0]/2)
            xd = self.magnitude*math.cos(self.orientation)
            yd = self.magnitude*math.sin(self.orientation)

            # This is where we set the pixel corresponding to the basis frequency to be 'lit' - to have amplitude
            # Use half the max energy of image, assuming brightness = 1.0
            # But, if you want to vary the brightness of the basis, scale it here.
            # First, let's set the 0th frequency offset around which all other frequencies fluctuate
            a[cy,cx] = self.im.shape[0]*self.im.shape[1] / 2.0
            a[int(cy+yd), int(cx+xd)] = self.im.shape[0]*self.im.shape[1] / 2.0
            amplitude = np.fft.fftshift(a)

            # If you want to add a phase shift for this basis function:
            p[int(cy+yd), int(cx+xd)] = self.phaseOffset # Do it here :)
            phase = np.fft.fftshift(p)

            # Note the reconstructed image (top right) as we light up different basis frequencies.
            '''
        
        # Part 2: Reconstructing from different numbers of basis frequencies
        # ==================================================================
        elif self.hw2part == 2:
            '''
            # In this part, we change the number of bases shown in the reconstruction of the original image. This is displayed as an animation

            # Make a square mask over the amplitude image
            Y, X = np.ogrid[:self.im.shape[0], :self.im.shape[1]]
            # Suppress frequencies less than cutoff distance
            mask = np.logical_or( np.abs(X-(self.im.shape[1]/2)) >= self.frequencyCutoffDist, np.abs(Y-(self.im.shape[0]/2)) >= self.frequencyCutoffDist )
            a = np.fft.fftshift(amplitude)
            a[mask] = 0
            amplitude = np.fft.fftshift(a)

            # Slowly undulate the cutoff radius back and forth
            # If radius is small and direction is decreasing, then flip the direction!
            if self.frequencyCutoffDist <= 1 and self.frequencyCutoffDirection < 0:
                self.frequencyCutoffDirection *= -1
            # If radius is large and direction is increasing, then flip the direction!
            if self.frequencyCutoffDist > self.im.shape[1]/3 and self.frequencyCutoffDirection > 0:
                self.frequencyCutoffDirection *= -1
            
            self.frequencyCutoffDist += self.frequencyCutoffDirection
            '''

        # Part 3: Replacing amplitude / phase with that of another image
        # ==============================================================
        elif self.hw2part == 3:    
            '''
            imMerlinFFT = np.fft.fft2( self.imMerlinTheCat )
            amplitudeMerlin = np.sqrt( np.power( imMerlinFFT.real, 2 ) + np.power( imMerlinFFT.imag, 2 ) )
            phaseMerlin = np.arctan2( imMerlinFFT.imag, imMerlinFFT.real )
            
            # Try uncommenting either of the lines below
            #amplitude = amplitudeMerlin
            phase = phaseMerlin
            '''

        # Part 4: Replacing amplitude / phase with that of a noisy image
        # ==============================================================
        elif self.hw2part == 4:
            '''
            # Generate some noise
            self.uniform_noise = np.random.uniform( 0, 1, self.im.shape )
            imNoiseFFT = np.fft.fft2( self.uniform_noise )
            amplitudeNoise = np.sqrt( np.power( imNoiseFFT.real, 2 ) + np.power( imNoiseFFT.imag, 2 ) )
            phaseNoise = np.arctan2( imNoiseFFT.imag, imNoiseFFT.real )
            
            # Try uncommenting either of the lines below
            amplitude = amplitudeNoise
            #phase = phaseNoise
            '''

        # Part 5: Understanding amplitude and phase
        # =========================================
        elif self.hw2part == 5:    
            '''
            # Play with the images. What can you discover? Try uncommenting each modification, one at a time, to see its direct image. Feel free to combine these modifications for different effects.
            
            # Zero out 0th infinite-frequency component? ('DC offset')
            amplitude[0,0] = 0

            # Zero out only phase?
            # phase = np.zeros( self.im.shape ) # + 0.5 * phase

            # Flip direction?
            # phase = -phase

            # Rotate all phase values?
            # self.phaseOffset += 0.05
            # phase += self.phaseOffset

            # Rotate whole image? Together? Individually?
            # phase = np.rot90(phase)
            # amplitude = np.rot90(amplitude)
            
            # Are these manipulations meaningful?
            # What other manipulations might we perform?
            '''
        
        return amplitude, phase

if __name__ == '__main__':
    live_FFT2()
