import cv2
# functions.py : Monaco Francesco Pio
def selective_search(image, method="fast"):
    '''
    Parameters
    -----
    image : cv.Mat
      image where to find the boxes
    method : string
      type of analysis

    Returns
    ------
    list
      a vector with bounding boxes
    '''
    # initialize OpenCV's selective search implementation and set the
    # input image
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    # check to see if we are using the fast but less accurate version
    # of selective search
    if method == "fast":
        ss.switchToSelectiveSearchFast()
    # otherwise we are using the slower but more accurate version
    else:
        ss.switchToSelectiveSearchQuality()
    # run selective search on the input image
    rects = ss.process()
    # return the region proposal bounding boxes
    return rects

def canny_len(image):
  '''
  Parameters
  -------
  image : cv.Mat
    Image to analyze

  Returns
  ------
  int
    y-length of the objects
    x-length of the objects
  '''
  image_gray =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  canny_image = cv2.Canny(image_gray, threshold1=100, threshold2=210)
  contours, _ = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Find the contour with the longest length
  longest_contour = max(contours, key=lambda contour: len(contour))

  # Get the y length
  _, y, w, h = cv2.boundingRect(longest_contour)
  return h, w