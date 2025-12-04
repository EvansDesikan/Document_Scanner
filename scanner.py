import cv2
import numpy as np

# --- FUNCTION 1: Order the 4 corners of the paper ---
def order_points(pts):
    # Initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left
    rect[2] = pts[np.argmax(s)] # Bottom-right

    # Now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-right
    rect[3] = pts[np.argmax(diff)] # Bottom-left

    return rect

# --- FUNCTION 2: The Actual Perspective Transform ---
def four_point_transform(image, pts):
    # 1. Obtain a consistent order of the points
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 2. Calculate the Width of the new image
    # Distance between bottom-right and bottom-left
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    # Distance between top-right and top-left
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 3. Calculate the Height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 4. Construct the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, specifying points in the top-left,
    # top-right, bottom-right, and bottom-left order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 5. Compute the Perspective Transform Matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

# --- MAIN CODE STARTS HERE ---

# 1. READ IMAGE
image = cv2.imread('paper2.jpg')
# Resize for easier processing (optional, but good for speed)
image = cv2.resize(image, (600, 800))
orig = image.copy() # Keep a copy of the original

# 2. PREPROCESSING
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 75, 200)

# 3. FIND CONTOURS
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

doc_contour = None

# Loop to find the 4-point rectangle
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        doc_contour = approx
        break

# 4. APPLY TRANSFORM & SHOW RESULT
if doc_contour is not None:
    # We need to reshape the contour to just get the 4 points without extra nesting
    pts = doc_contour.reshape(4, 2)
    
    # Run the magic transform function
    warped = four_point_transform(orig, pts)

    # Convert to grayscale
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # --- IMPROVED THRESHOLDING ---
    # 1. Increase the block size (11 -> 21) to look at larger areas
    # 2. Increase the C constant (2 -> 10) to filter out background noise
    warped_bin = cv2.adaptiveThreshold(warped_gray, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 21, 10)

    # Show results
    cv2.imshow("Original", image)
    cv2.imshow("Scanned (Cleaned)", warped_bin) # Should look much cleaner now
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # SAVE THE RESULT
    cv2.imwrite("scanned_document.jpg", warped_bin)
    print("Success! Image saved as scanned_document.jpg")

else:
    print("Could not find 4 points. Try a clearer background.")