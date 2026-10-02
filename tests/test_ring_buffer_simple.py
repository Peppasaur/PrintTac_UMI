from reactive_diffusion_policy.common.ring_buffer import RingBuffer

def test_ring_buffer():
    # Example usage
    buffer = RingBuffer(size=5)

    # Push some items
    for i in range(10):
        buffer.push(f"item_{i}")

    # Get last 3 items
    items, err = buffer.peek_last_n(3)
    if err is None:
        print("Last 3 items:", items)  # Will print newest to oldest
        # Expected: ['item_9', 'item_8', 'item_7']

if __name__ == "__main__":
    test_ring_buffer()
