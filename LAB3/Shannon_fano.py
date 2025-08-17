import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Shannon-Fano Encoding Function
def shannon_fano(symbols):
    if len(symbols) <= 1:
        return {symbols[0][0]: "0"} if symbols else {}
    total = sum([s[1] for s in symbols])
    acc = 0
    for i in range(len(symbols)):
        acc += symbols[i][1]
        if acc >= total/2:
            break
    left = shannon_fano(symbols[:i+1])
    right = shannon_fano(symbols[i+1:])
    left = {k: '0'+v for k,v in left.items()}
    right = {k: '1'+v for k,v in right.items()}
    return {**left, **right}

# Read grayscale image
img = cv2.imread("download.jpg", cv2.IMREAD_GRAYSCALE)

# Get frequency of pixels
freq = Counter(img.flatten())
symbols = sorted(freq.items(), key=lambda x: x[1], reverse=True)

# Build Shannon-Fano codebook
codebook = shannon_fano(symbols)

# Encode image
encoded = ''.join([codebook[p] for p in img.flatten()])

# Decode back
decode_dict = {v: k for k, v in codebook.items()}
decoded = []
buffer = ""
for bit in encoded:
    buffer += bit
    if buffer in decode_dict:
        decoded.append(decode_dict[buffer])
        buffer = ""
decoded_img = np.array(decoded, dtype=np.uint8).reshape(img.shape)

# Show images
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Decoded Image")
plt.imshow(decoded_img, cmap="gray")
plt.axis("off")

plt.show()
