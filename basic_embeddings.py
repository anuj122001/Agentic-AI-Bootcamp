# Loads the SentenceTransformer class, used to convert text into embedding vectors
from sentence_transformers import SentenceTransformer
# Loads PCA, used to reduce high-dimensional vectors down to 2D for plotting
from sklearn.decomposition import PCA
# Numpy, used here for the dot product (cosine similarity) calculation
import numpy as np
# Matplotlib, used to draw the 2D scatter plot
import matplotlib.pyplot as plt

# Load a small pretrained embedding model; it outputs 384-dimensional vectors per input
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
# List of words/phrases we want to embed and compare
sentences = [
  "Dog",
  "Puppy",
  "Cat",
  "Kitten",
  "Cricket",
  "Wicket-keeper",
]
# Convert each sentence into a 384-dim vector; normalize_embeddings=True makes each vector unit length
# so that a simple dot product equals cosine similarity. E has shape (6, 384)
E = model.encode(sentences, normalize_embeddings=True)  # shape: (n, 384)

# 2D for plotting (PCA)
# Reduce the 384 dimensions down to 2 dimensions (the 2 directions with most variance) so we can plot them
xy = PCA(n_components=2).fit_transform(E)
# Loop through each sentence and its corresponding (x, y) 2D point
for s, (x, y) in zip(sentences, xy):
    # Print the coordinates and the word, formatted with a sign and 2 decimal places
    print(f"{x:+.2f}\t{y:+.2f}\t{s}")


# Plot the embeddings
# Create a new figure (chart canvas) sized 10x8 inches
plt.figure(figsize=(10, 8))
# Draw all 6 points as blue dots on the chart
plt.scatter(xy[:, 0], xy[:, 1], c='blue', s=100, alpha=0.7)

# Add labels for each point
# Loop through each sentence with its index so we can label the matching point
for i, sentence in enumerate(sentences):
    # Write the word as a text label next to its point, offset by (5,5) pixels so it doesn't overlap the dot
    plt.annotate(sentence, (xy[i, 0], xy[i, 1]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=12, fontweight='bold')

# Label the x-axis
plt.xlabel('First Principal Component')
# Label the y-axis
plt.ylabel('Second Principal Component')
# Set the chart title
plt.title('Word Embeddings Visualization (PCA)')
# Add a light grid in the background for readability
plt.grid(True, alpha=0.3)
# Adjust spacing so labels/titles don't get cut off
plt.tight_layout()
# Display the chart in a window
plt.show()

# Cosine similarity example
# Since embeddings are unit-length (normalized), the dot product of two vectors equals their cosine similarity
def cos(a,b): return float(np.dot(a,b))
# Similarity between "Dog" and "Puppy" — expected to be high (related concepts)
print(cos(E[0], E[1]))
# Similarity between "Dog" and "Cat" — expected to be moderately high (both animals)
print(cos(E[0], E[2]))
# Similarity between "Dog" and "Cricket" — expected to be low (unrelated concepts)
print(cos(E[0], E[4]))

