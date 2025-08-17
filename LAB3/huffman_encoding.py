import cv2
import numpy as np
import heapq
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------- Huffman Encoding Helpers ----------
class Node:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
    def __lt__(self, other):  # for heapq
        return self.freq < other.freq

def build_huffman_tree(frequencies):
    heap = [Node(freq, sym) for sym, freq in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        merged = Node(n1.freq+n2.freq, left=n1, right=n2)
        heapq.heappush(heap, merged)
    return heap[0]

def build_codes(node, prefix="", codebook={}):
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    else:
        build_codes(node.left, prefix+"0", codebook)
        build_codes(node.right, prefix+"1", codebook)
    return codebook

# ---------- Main Process ----------
# Read grayscale image
img = cv2.imread("download.jpg", cv2.IMREAD_GRAYSCALE)

# Count frequencies
unique, counts = np.unique(img, return_counts=True)
frequencies = dict(zip(unique, counts))

# Build Huffman codes
root = build_huffman_tree(frequencies)
codes = build_codes(root)

# Encode image
encoded_data = ''.join(codes[pixel] for row in img for pixel in row)

# Decode image
decoded_pixels = []
current_code = ""
for bit in encoded_data:
    current_code += bit
    for sym, code in codes.items():
        if code == current_code:
            decoded_pixels.append(sym)
            current_code = ""
            break

decoded_img = np.array(decoded_pixels, dtype=np.uint8).reshape(img.shape)

# ---------- Show Results ----------
plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.imshow(img, cmap='gray'); plt.title("Original")
plt.subplot(1,2,2); plt.imshow(decoded_img, cmap='gray'); plt.title("Decoded")
plt.show()

print("Original size (bits):", img.size * 8)
print("Compressed size (bits):", len(encoded_data))
