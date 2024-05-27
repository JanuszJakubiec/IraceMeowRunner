defmodule Schaffer do
  import Nx.Defn
  import Tuple

  @size 10

  @two_pi 2 * :math.pi()
  defn evaluate() do
    genomes = Nx.tensor([[1,1,1,1,1,1,1,1,1,1], [1,2,3,4,5,6,7,8,9,10], [0,0,0,0,0,0,0,0,0,0]], type: :f64)

    {head, _} = Nx.shape(genomes)

    sums =
      1/(head-1) * Nx.sum(Nx.sqrt(s(genomes)) + Nx.sqrt(s(genomes)) * Nx.pow(Nx.sin(50*Nx.pow(s(genomes),0.2)),2),axes: [1])
      |> Nx.pow(2)

    -sums
  end

  defn s(genomes) do
    {head, _}  = Nx.shape(genomes)
    si=
      (Nx.pow(Nx.slice(genomes,[0,1], [head, @size-1]),2) + Nx.pow(Nx.slice(genomes,[0,0], [head, @size-1]),2))
      |> Nx.sqrt()
    si

  end
end
