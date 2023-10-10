defmodule Clique do
  import Nx.Defn

  @size 10

  defn evaluate(_genomes) do
    edges = Nx.tensor([[1, 3], [1, 2], [2, 3], [1, 4], [2, 4], [4, 5], [3, 5], [5, 6], [1, 5], [2, 5], [0,0]])
    v = 7
    genomes = Nx.tensor([[1,1,1,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,1,1,1], [1,0,1,0,0,0,0,0,0,0]])

    {length, _} = Nx.shape(genomes)

    matrix = Nx.broadcast(0, {v, v})
    ones = Nx.broadcast(1, {v, v})

    xyz = Nx.iota({7,2}, axis: 1)
          |> Nx.transpose(axes: [1, 0])


    indices = Nx.select(genomes, Nx.iota({@size}), Nx.tensor(10))

    taken_edges = Nx.take(edges, indices)
    indices = Nx.reshape(taken_edges, {3, 2*10})
    eye = Nx.eye(v)
    markings = Nx.take(eye, indices)
    trans = Nx.transpose(markings, axes: [0, 2, 1])
    vertices = Nx.sum(trans, axes: [2])
    |> Nx.add(Nx.iota({v}) < 1)
    indexes_not_present = Nx.select(vertices > 0, 0, Nx.iota({v}))
    |> Nx.add(Nx.iota({v}) < 1)

    whatIWant = Nx.take(xyz, indexes_not_present > 0)
    whatIWant2 = Nx.transpose(whatIWant, axes: [0,2,1])

    final = Nx.add(whatIWant, whatIWant2) > 0
    first = Nx.iota({length, @size, 1}, axis: 0)
    taken_edges2 = Nx.reverse(taken_edges, axes: [2])
    taken_edges2 = Nx.concatenate([first, taken_edges2], axis: 2)
    |> Nx.flatten(axes: [0,1])
    taken_edges = Nx.concatenate([first, taken_edges], axis: 2)
    |> Nx.flatten(axes: [0,1])
    result = Nx.indexed_add(final, taken_edges, Nx.broadcast(1, {@size*length}))
    |> Nx.indexed_add(taken_edges2, Nx.broadcast(1, {@size*length}))
    |> Nx.add(Nx.eye(7)) > 0
    res = Nx.reduce(result, 1, [axes: [1,2]], fn x,y -> Nx.logical_and(x,y) end)
    clique_possible_size = Nx.sum(vertices > 0, axes: [1]) |> Nx.add(-1)
    Nx.select(res, clique_possible_size, 0)
  end
end
