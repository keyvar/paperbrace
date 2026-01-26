import numpy as np


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """
    L2-normalize float32 vectors.

    Accepts either:
      - shape (N, dim): returns row-wise normalized vectors
      - shape (dim,): returns a normalized vector

    Uses a small epsilon to avoid divide-by-zero.
    """
    v = np.asarray(v, dtype=np.float32)

    if v.ndim == 1:
        denom = np.linalg.norm(v) + 1e-12
        return v / denom

    denom = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / denom



class Distance:
    """
    Distance metrics for ranking vector search results.

    These methods are designed so that, for a fixed query vector and a fixed set of
    candidate embeddings, sorting by the returned distances (ascending) yields the
    same ordering as ChromaDB using the corresponding space (cosine / inner-product / L2).
    However, the exact numeric values may differ from ChromaDB because of normalization, 
    but guaranteeing the ordering is the primary goal.
    For all returned distances, lower means better.
    """

    @staticmethod
    def cosine(emb: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Cosine distance in a Chroma/HNSWlib like form.

        Assumptions:
        - emb has shape (N, dim)
        - q has shape (dim,)
        - emb rows and q are L2-normalized so that (emb @ q) equals cosine similarity.
        - The exact value is different than ChromaDB becasue the latter does not normalize the embeddings.

        Returns:
        - distances of shape (N,), lower is better.
        """
        # emb @ q is similarity, the distance is 1 - similarity
        return 1.0 - emb @ q # distance

    @staticmethod
    def inner_product(emb: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Inner-product distance in a Chroma/HNSWlib like form.

        Assumptions:
        - Same as _cosine method.

        Returns:
        - distances of shape (N,), lower is better.
        """
        
        return 1.0 - emb @ q

    @staticmethod
    def l2_squared(emb: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Squared Euclidean distance in a Chroma/HNSWlib like form.

        Many ANN/vector backends use squared L2 internally. Squared L2 preserves
        the same ordering as true L2 while being cheaper to compute (no sqrt).

        Assumptions:
        - Same as _cosine method.

        Returns:
        - distances of shape (N,), lower is better.
        """
        diff = emb - q
        return np.einsum("ij,ij->i", diff, diff)
