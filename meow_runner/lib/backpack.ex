defmodule Backpack do
  import Nx.Defn

  @weights Nx.tensor([100, 50, 20, 30, 10, 50, 40, 20, 60, 10], type: {:u, 64})
  @max_weight 100
  @values Nx.tensor([5, 100, 50, 10, 10000, 10, 5, 50, 50, 10], type: {:u, 64})

  defn evaluate_backpack(genomes) do
    #genomes = Nx.tensor([[0, 0, 1, 1, 0,, [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]0, 0, 0, 1, 1, 0, 0],[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],[0, 0, 0, 0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],[0, 0, 0, 0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]], type: {:u, 64})
    weights = Nx.select(genomes, @weights, Nx.tensor(0))
    value = Nx.select(genomes, @values, Nx.tensor(0))
    weights_sum = Nx.sum(weights, axes: [1])
    value_sum = Nx.sum(value, axes: [1])
    Nx.select(weights_sum <= @max_weight, value_sum, -weights_sum)
  end
end
