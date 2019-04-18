########################################################################
#
# File:   p1.py
# Author: Ivan Lomeli
# Date:   February 2019
#
# Written for ENGR 27 - Computer Vision
#
########################################################################
#
# This program demonstrates how to perform thresholding and morphological
# operators to video frames in order to perform blob tracking.
#
# Usage: the program can be run with a filename as a command line argument
# a command line argument.  If no argument is given, tries to capture from the
# default input 'vid1.mov'
"""
Created on Sat Feb 16 15:09:53 2019


"""

from __future__ import print_function

import cv2
import numpy
import sys

def main():
    # Figure out what input we should load:
    if len(sys.argv) > 1:
        input_filename = sys.argv[1]
    else:
        print('Using default input \'vid1.mov\'')
        print()
        print('  python', sys.argv[0])
        print()
        input_filename = 'vid1.mov'

    capture = cv2.VideoCapture(input_filename)
    if capture:
            print('Opened file', input_filename)
    # Bail if error.
    if not capture or not capture.isOpened():
        print('Error opening video!')
        sys.exit(1)

    # Fetch the first frame and bail if none.
    ok, frame = capture.read()

    if not ok or frame is None:
        print('No frames in video')
        sys.exit(1)
    # We will use information from initial frame to set up a VideoWriter to
    # output video.
    w = frame.shape[1]
    h = frame.shape[0]

    # Make a recorder for the (x,y) coordinates of a single moving object
    centerList = []
    thresList1=[]
    trajectoryList=[]

    # Define the lower and upper boundaries of the "green" ball in the HSV color
    #  space (Range for OpenCV is H: 0-180, S:0-255, V:0-255)
    orangeLower = numpy.array([5,50,50],numpy.uint8)
    orangeUpper = numpy.array([15,255,255],numpy.uint8)

    print("Starting to read frames...")
    # Loop until video is ended
    while True:
        # fetch next frame
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        # threshold frame image
        frame1=thresholdColor(frame, orangeLower, orangeUpper)
        # Perform a series of dilations and erosions to remove any small blobs
        # left in the mask
        close = cv2.dilate(frame1, None, iterations=10)
        height, width, _ = frame.shape
        assert height == h
        assert width == w
        if close is not None:
            thresList1.append(close)
        # Finds contours
        withContours,centerList = regions(close,centerList)
        if len(centerList)>1:
            arrayCenterList= numpy.array(centerList).reshape((-1,1,2)).astype(numpy.int32)
            cv2.drawContours(withContours, [arrayCenterList], 0, (0, 255, 0), 2)
        trajectoryList.append(withContours)
    print("Finished reading frames!\n")

    cv2.destroyAllWindows()

    # Save and play video results
    thresAndMorp = writeFile(thresList1,"thresAndMorp.",w,h,10,False)
    trajectory = writeFile(trajectoryList,"trajectory.",w,h,10,True)
    print("Now, play the video after threshold and morphological operation applied...")
    play(thresAndMorp)
    obtainFrames(thresList1, "TreshMorp")

    print("Now, play the video with contours and trajectory...")
    play(trajectory)
    obtainFrames(trajectoryList, "Traject")

def obtainFrames(frames, label):
    counterNo = 0
    sixth = len(frames)/6
    twoSixths = sixth*2
    half = sixth*3
    fourSixths = sixth*4
    fiveSixths = sixth*5
    done = len(frames)

    for frame in frames:
        counter = '%d' % (counterNo)
        if counterNo == 0:
            cv2.imwrite('frame'+counter+label+'.png', frame)
        if counterNo == sixth:
            cv2.imwrite('frame'+counter+label+'.png', frame)
        if counter == twoSixths:
            cv2.imwrite('frame'+counter+label+'.png', frame)
        if counterNo == half:
            cv2.imwrite('frame'+counter+label+'.png', frame)
        if counterNo == 44 and len(frame) > 44:
            cv2.imwrite('frame'+counter+label+'.png', frame)
        if counter == fourSixths:
            cv2.imwrite('frame'+counter+label+'.png', frame)
        if counterNo == 55 and len(frame) > 55:
            cv2.imwrite('frame'+counter+label+'.png', frame)
        if counterNo == 60 and len(frame) > 60:
            cv2.imwrite('frame'+counter+label+'.png', frame)
        if counterNo == 65 and len(frame) > 65:
            cv2.imwrite('frame'+counter+label+'.png', frame)
        if counterNo == 68 and len(frame) > 68:
            cv2.imwrite('frame'+counter+label+'.png', frame)
        if counterNo == fiveSixths:
            cv2.imwrite('frame'+counter+label+'.png', frame)
        if counterNo == done-1:
            cv2.imwrite('frame'+counter+label+'.png', frame)
        counterNo+=1

def noLabelSave(image, text):
    cv2.imwrite(text+'.png', image)

def writeFile(frameList,filename,w,h,fps,isColor):
    # Beginning setting up VideoWriters to output video.
    print("Setting up VideoWriters...")
    #fourcc, ext = (cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 'avi')
    fourcc, ext = (cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 'mov')
    filename = filename+ext
    writerObject = cv2.VideoWriter(filename, fourcc,fps, (w, h),isColor)
    print("Finished setting up VideoWriters!")
    for frame in frameList:
        writerObject.write(frame)
    cv2.destroyAllWindows()
    writerObject.release()
    return filename

def thresholdColor(frame, min, max):
    # blur the frame and convert it to the HSV color space
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# construct a mask for the color orange
    mask = cv2.inRange(hsv, min, max)
    return mask


#---CONTOUR--this part is a slightly modified version of Matt's regions.py
######################################################################
# A list of RGB colors useful for drawing segmentations of binary
# images with cv2.drawContours

CONTOUR_COLORS = [
    (255,   0,   0),
    (255,  63,   0),
    (255, 127,   0),
    (255, 191,   0),
    (255, 255,   0),
    (191, 255,   0),
    ( 63, 255,   0),
    (  0, 255,   0),
    (  0, 255,  63),
    (  0, 255, 127),
    (  0, 255, 191),
    (  0, 255, 255),
    (  0, 191, 255),
    (  0, 127, 255),
    (  0,  63, 255),
    (  0,   0, 255),
    ( 63,   0, 255),
    (127,   0, 255),
    (191,   0, 255),
    (255,   0, 255),
    (255,   0, 191),
    (255,   0, 127),
    (255,   0,  63),
]


######################################################################
# Compute moments and derived quantities such as mean, area, and
# basis vectors from a contour as returned by cv2.findContours.
# Feel free to use this function in your project 2 code.
#
# Returns a dictionary.

def getcontourinfo(c):
    m = cv2.moments(c)

    s00 = m['m00']
    s10 = m['m10']
    s01 = m['m01']
    c20 = m['mu20']
    c11 = m['mu11']
    c02 = m['mu02']

    if s00 != 0:
        #centroid
        mx = s10 / s00
        my = s01 / s00

        A = numpy.array( [
                [ c20 / s00 , c11 / s00 ],
                [ c11 / s00 , c02 / s00 ]
                ] )

        W, U, Vt = cv2.SVDecomp(A)

        ul = 2 * numpy.sqrt(W[0,0])
        vl = 2 * numpy.sqrt(W[1,0])

        ux = ul * U[0, 0]
        uy = ul * U[1, 0]

        vx = vl * U[0, 1]
        vy = vl * U[1, 1]

        mean = numpy.array([mx, my])
        uvec = numpy.array([ux, uy])
        vvec = numpy.array([vx, vy])


    else:
        mx=s10
        my=s01
        mean = c[0].astype('float')
        uvec = numpy.array([1.0, 0.0])
        vvec = numpy.array([0.0, 1.0])

    return {'moments': m,
            'area': s00,
            'mean': mean,
            'b1': uvec,
            'b2': vvec,
            'center':(mx,my)}

######################################################################
# Construct a tuple of ints from a numpy array

def make_point(arr):
    return tuple(numpy.round(arr).astype(int).flatten())

######################################################################

def regions(image,centerList):

    work = image.copy()

    # Create an RGB display image which to show the different regions.
    display = numpy.zeros((image.shape[0], image.shape[1], 3),
                          dtype='uint8')

    # Get the list of contours in the image. See OpenCV docs for
    # information about the arguments.
    image, contours, hierarchy = cv2.findContours(work, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #print('found', len(contours), 'contours')

    # Loop through to draw contours:
    for j in range(len(contours)):

        # Choose a color
        u = 1
        i = int(round(u * (len(CONTOUR_COLORS)-1)))

        # Draw the contour as a colored region on the display image.
        cv2.drawContours( display, contours, j, CONTOUR_COLORS[i], -1 )

    # Define the colors black & white (see below)
    white = (255,255,255)
    black = (0, 0, 0)

    # Loop through again to draw labels
    for contour in contours:

        # Compute some statistics about this contour.
        info = getcontourinfo(contour)

        # Mean location, area, and basis vectors can be useful.
        area = info['area']
        mu = info['mean']
        b1 = info['b1']
        b2 = info['b2']
        center = info['center']
        centerList.append(center)

        for width, color in [(3, black), (1, white)]:

            # Annotate the display image with mean and basis vectors.
            cv2.circle( display, make_point(mu), 3, color,
                        width, cv2.LINE_AA )

            cv2.line( display, make_point(mu), make_point(mu+b1),
                      color, width, cv2.LINE_AA )

            cv2.line( display, make_point(mu), make_point(mu+b2),
                      color, width, cv2.LINE_AA )

        for width, color in [(3, black), (1, white)]:

            cv2.putText( display, 'Area: {:.0f} px'.format(area),
                         make_point(mu + (-5, -10)),
                         cv2.FONT_HERSHEY_PLAIN,
                         0.8, color, width, cv2.LINE_AA )
    return display,centerList

"""
To play the videos
"""
def play(filename):
    capture = cv2.VideoCapture(filename)
    ok, frame = capture.read()
    while True:
        if not ok or frame is None:
            break
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        ret, frame = capture.read()
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
