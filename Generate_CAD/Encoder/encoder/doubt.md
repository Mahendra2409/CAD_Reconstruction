# Doubt List

1.  **Function:** `_compute_closeness_core`

    **Code:** `'''nodes which are closer to each other have smaller direct edge distance value in adjacency matrix,`
2.  **Function:** `LearnableCentralityEncoding.__init__`

    **Code:** ` This says that these are constants and will not change, used for optimization`
3.  **Function:** `LearnableCentralityEncoding.forward`

    **Code:** ` !!! Its value depend on only first batch, Diferernt batches have different shapes of graph),`
4.  **Function:** `LearnableCentralityEncoding.forward`

    **Code:** ` so closeness_centrality will be different but here it is using clossness_centrality of first batch only`
5.  **Function:** `CentralityEncoding.forward`

    **Code:** `'''`
6.  **Function:** `CentralityEncoding.forward`

    **Code:** `'''`
7.  **Function:** `SpatialEncoding.forward`

    **Code:** `!!!max_path_distance is not used in this function`
8.  **Function:** `SpatialEncoding.forward`

    **Code:** ` !!! direct edge weights, it only contain that edges into the weight_matrix which are between adjacent nodes`
9.  **Function:** `EdgeEncoding.__init__`

    **Code:** ` !!! why (1, edge_dim)?? why not (max_path_distance, edge_dim), edge vector for each position in the path?`
10. **Function:** `EdgeEncoding.__init__`

    **Code:** ` Its not learning the relation between the edges along path, and can not implementable on forward pass else loop`
11. **Function:** `EdgeEncoding.forward`

    **Code:** `!!! Is weights coming from (shortest_path_distance) or (batched_shortest_path_distance) only?`
12. **Function:** `EdgeEncoding.forward`

    **Code:** ` !!! mean() will be claculated first and then multiplied with the (scaled_weights)`
13.  **Function:** `EdgeEncoding.forward`

    **Code:** `!!! There is problem in edge_vector and weights initialization (They are always not suitable for this loop), here it required same as mentioned in the paper`
14.  **Function:** `GraphormerAttentionHead.forward`

    **Code:** ` why c.sum(-1),, Its doing (num_nodes, num_nodes) -> (num_nodes)?? In Graphformer paper, it is not mentioned`
15.  **Function:** `GraphormerAttentionHead.forward`

    **Code:** ` Mask is not applied to a and b , this will lead to nodes of a graph are influenced by nodes of other graphs (in a and b)`
16.  **Function:** `GraphormerMultiHeadAttention.forward`

    **Code:** `!!! It assigning same input(x, edge_attr, b, weights, ptr) with full dimension to all heads, Every head watches the same input`
17.  **Function:** `GraphormerMultiHeadAttention.forward`

    **Code:** `(num_nodes, num_heads * dim_k) mostprobably dim_k = num_nodes ?`
