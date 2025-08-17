import heapq

def huffman(symbols_probs):
    # Step 1: Create priority queue (min-heap)
    heap = [[prob, [symbol, ""]] for symbol, prob in symbols_probs.items()]
    heapq.heapify(heap)

    # Step 2: Merge nodes until one tree remains
    while len(heap) > 1:
        lo = heapq.heappop(heap)  # least prob
        hi = heapq.heappop(heap)  # second least prob
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    # Step 3: Return codes
    return dict(sorted(heap[0][1:], key=lambda x: x[0]))


# Example usage
symbols_probs = {
    'A': 0.4,
    'B': 0.3,
    'C': 0.2,
    'D': 0.1
}

huff_codes = huffman(symbols_probs)
print("\nHuffman Codes:")
for symbol, code in huff_codes.items():
    print(symbol, ":", code)
