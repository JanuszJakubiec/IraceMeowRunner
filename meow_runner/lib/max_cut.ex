defmodule MaxCut do
  import Nx.Defn

  defn evaluate(genomes) do
    # genomes = Nx.tensor([[0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]])

    # Define a tensor of edges in the graph, where each row
    # represents an edge between two vertices
    edges =
      Nx.tensor([
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [1, 2],
        [1, 3],
        [5, 1],
        [5, 4]
      ])

    # Calculate the number of edges and genomes for later use
    {edges_n, _} = Nx.shape(edges)
    {genomes_n, _} = Nx.shape(genomes)

    # Transforming edges list to genomes_n flat tensors
    transfomed_edges =
      Nx.reshape(edges, {edges_n * 2, 1})
      |> Nx.broadcast({genomes_n, edges_n * 2, 1})

    genome_idx = Nx.iota({genomes_n, edges_n * 2, 1}, axis: 0)

    # Creating the flat list of indices that will be taken from genomes
    # To determine whether given vertex is assigned to class 0 or 1
    indices =
      Nx.concatenate([genome_idx, transfomed_edges], axis: 2)
      |> Nx.flatten(axes: [0, 1])

    # markers is a tensor that for each vertex in each edge determines, if
    # it belongs to group 0 or 1.
    markers =
      Nx.gather(genomes, indices)
      |> Nx.reshape({edges_n * genomes_n, 2})
      |> Nx.transpose(axes: [1, 0])

    # Evaluate the cut made by each genome's vertex assignments by
    # computing the XOR of vertex group memberships
    # across each edge, then summing the total number of edges cut for each genome
    Nx.logical_xor(markers[0], markers[1])
    |> Nx.reshape({genomes_n, edges_n})
    |> Nx.sum(axes: [1])
  end
end
