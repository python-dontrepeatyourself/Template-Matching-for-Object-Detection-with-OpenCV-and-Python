# import the necessary packages
import cv2


# load the main image and the template image
image = cv2.imread("examples/1.jpg")
template = cv2.imread("examples/template1.jpg")
# make a copy of the image
image_copy = image.copy()

# convert the images to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# get the width and height of the template image
template_h, template_w = template.shape[:-1]

# perform template matching using the normalized cross-correlation method
result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)

# find the location of the best match in the result map
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# draw a rectangle around the best match
top_left = max_loc
bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
cv2.rectangle(image_copy, top_left, bottom_right, (0, 255, 0), 2)

# show the images
cv2.imshow("Image", image)
cv2.imshow("Template", template)
cv2.imshow("Matched Template", image_copy)
cv2.waitKey(0)
