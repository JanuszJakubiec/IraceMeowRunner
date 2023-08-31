defmodule Rastrigin do
  import Nx.Defn

  @size 10

  @two_pi 2 * :math.pi()

  defn evaluate_rastrigin(genomes) do
    sums =
      (@size + Nx.pow(genomes, 2) - @size * Nx.cos(genomes * @two_pi))
      |> Nx.sum(axes: [1])

    -sums
  end
end
