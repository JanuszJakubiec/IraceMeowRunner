defmodule NQueen do
  import Nx.Defn

  defn evaluate(_genomes) do
    size = 8

    genomes =
      Nx.tensor([
        [
          [1, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 1]
        ],
        [
          [1, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 0]
        ]
      ])

    genomes = genomes |> Nx.flatten(axes: [1, 2])

    {genomes_n, _} = Nx.shape(genomes)

    genomes = genomes |> Nx.reshape({genomes_n, size, size})

    # |> Nx.slice([0, 0, 0], [genomes_n, size, size * 2], strides: [1, 1, 1])

    # t = Nx.iota({900})
    # t = Nx.reshape(t, {2, 15, 30})
    # Nx.slice(t, [0, 4, 11], [2, 3, 9], strides: [1, 1, 1])
    Nx.take_diagonal(genomes[0])
  end
end
