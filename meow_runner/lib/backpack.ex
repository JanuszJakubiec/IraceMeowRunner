defmodule Backpack do
  import Nx.Defn

  @weights Nx.tensor([100, 50, 20, 30, 10, 50, 40, 20, 60, 10], type: {:u, 64})
  @max_weight 100
  @values Nx.tensor([5, 100, 50, 10, 10000, 10, 5, 50, 50, 10], type: {:u, 64})

  defn evaluate(genomes) do
    # genomes = Nx.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],[0, 0, 0, 0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],[0, 0, 0, 0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]], type: {:u, 64})

    weights_sum =
      genomes
      |> Nx.select(@weights, Nx.tensor(0))
      |> Nx.sum(axes: [1])

    value_sum =
      genomes
      |> Nx.select(@values, Nx.tensor(0))
      |> Nx.sum(axes: [1])

    Nx.select(weights_sum <= @max_weight, value_sum, -weights_sum)
  end
end
