from chunking import chunk_text

text = """
Paris was amazing. We visited the Eiffel Tower,
Louvre Museum, cafes and many historical places.
The food was excellent and the nightlife was great.
Then we travelled to Rome and discussed Redis caching.
Finally we planned a Tokyo sushi trip.
"""

chunks = chunk_text(
    text,
    chunk_size=80,
    overlap=20,
)

for i, chunk in enumerate(chunks):

    print(f"\n--- CHUNK {i} ---")
    print(chunk)