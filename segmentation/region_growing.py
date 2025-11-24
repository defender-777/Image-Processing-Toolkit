import cv2
import numpy as np

def region_growing(image, seed, thresh=5):
    h, w = image.shape
    visited = np.zeros((h, w), dtype=bool)
    region = np.zeros((h, w), dtype=np.uint8)

    queue = [seed]
    seed_intensity = image[seed]

    while queue:
        x, y = queue.pop(0)
        if visited[x, y]:
            continue
        visited[x, y] = True

        if abs(int(image[x, y]) - int(seed_intensity)) < thresh:
            region[x, y] = 255

            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
                    queue.append((nx, ny))

    return region

# -------- RUN ----------

img = cv2.imread("image2.jpg", cv2.IMREAD_GRAYSCALE)
seed = (100, 150)       # Choose manually
mask = region_growing(img, seed, thresh=8)

cv2.imshow("Original", img)
cv2.imshow("Region Growing Result", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
