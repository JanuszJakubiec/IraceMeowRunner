defmodule Rastrigin do
  import Nx.Defn

  def size, do: 10

  @two_pi 2 * :math.pi()

  defn evaluate_rastrigin(genomes) do
    sums =
      (10 + Nx.pow(genomes, 2) - 10 * Nx.cos(genomes * @two_pi))
      |> Nx.sum(axes: [1])

    -sums
  end
end
