defmodule Clique do
  import Nx.Defn

  @genome_size 10

  defn evaluate(genomes) do
    vertices_n = 7
    genome_size = @genome_size

    edges =
      Nx.tensor([
        [1, 2],
        [1, 3],
        [1, 6],
        [2, 3],
        [2, 4],
        [2, 5],
        [3, 4],
        [3, 5],
        [4, 5],
        [5, 6],
        [0, 0]
      ])

    # genomes =
    #  Nx.tensor([
    #    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
    #  ])

    {genomes_n, _} = Nx.shape(genomes)

    edge_indices =
      Nx.select(genomes, Nx.iota({genome_size}), Nx.tensor(@genome_size))

    # Selecting edges from edges set, based on genomes
    taken_edges = Nx.take(edges, edge_indices)
    taken_edges_reversed = Nx.reverse(taken_edges, axes: [2])

    # Flattening edges list to single vertices
    vertices_from_edges = Nx.reshape(taken_edges, {genomes_n, 2 * genome_size})

    # Creating the marking tab, containing which vertices are present in each genome
    vertices =
      Nx.take(Nx.eye(vertices_n), vertices_from_edges)
      |> Nx.transpose(axes: [0, 2, 1])
      |> Nx.sum(axes: [2])
      |> Nx.add(Nx.iota({vertices_n}) < 1)

    ## For each genome, calculates the possible clique size(number of unique vertices used in genome)
    clique_possible_size = Nx.sum(vertices > 0, axes: [1]) |> Nx.add(-1)

    ### Creating the list with vertices that are not present in a genome. If the vertice i is present,
    # the value on position i is 0. When it is present the value on position i is i.
    vertices_not_present =
      Nx.select(vertices > 0, 0, Nx.iota({vertices_n}))
      |> Nx.add(Nx.iota({vertices_n}) < 1)
      |> Nx.greater(0)
      |> Nx.select(
        Nx.broadcast(1, {genomes_n, vertices_n}),
        Nx.broadcast(0, {genomes_n, vertices_n})
      )

    # Creating connections matrix, adding a row of ones, when vertice is not present
    connections_matrix = Nx.take(Nx.iota({2, vertices_n}, axis: 0), vertices_not_present)
    # Creating transposed connections matrix, adding a row of ones, when vertice is not present
    connections_matrix_tr = Nx.transpose(connections_matrix, axes: [0, 2, 1])

    # For genome containing vertices: [0, 1, 1] it will be equal to:
    # Summing connections matrix and transposed connections matrix. The remaining matrix will
    # contain 0 value, where the edge is needed for the genome to be a clique
    # 1 1 1 1
    # 1 1 1 1
    # 1 1 0 0
    # 1 1 0 0
    connections_matrix = Nx.add(connections_matrix, connections_matrix_tr) > 0

    genome_index =
      Nx.iota({genomes_n, genome_size * 2, 1}, axis: 0)
      |> Nx.flatten(axes: [0, 1])

    taken_edges =
      Nx.concatenate([taken_edges, taken_edges_reversed], axis: 1)
      |> Nx.flatten(axes: [0, 1])

    # Tensor, containing the 3-element lists, wiht [genome_id, edge_v_1, edge_v_2] it will be used,
    # to insert 1 into connections_matrix
    taken_edges =
      Nx.concatenate([genome_index, taken_edges], axis: 1)

    # Udating connections_matrix accordingly to taken_edges. Then if all fields in connections_matrix
    # are 1, the clique_possible_size is returned, 0 in other case
    Nx.indexed_add(
      connections_matrix,
      taken_edges,
      Nx.broadcast(1, {genome_size * 2 * genomes_n})
    )
    |> Nx.add(Nx.eye(vertices_n))
    |> Nx.greater(0)
    |> Nx.reduce(1, [axes: [1, 2]], fn x, y -> Nx.logical_and(x, y) end)
    |> Nx.select(clique_possible_size, 0)
  end

  defn return_value(genomes, val) do
    Nx.add(val, 1)
    {genomes_n, _} = Nx.shape(genomes)
    Nx.broadcast(1, {genomes_n})
  end
end
