defmodule SumSquares do
  import Nx.Defn

  @size 10

  defn evaluate(genomes) do
    #genomes = Nx.tensor([[1,1,1,1,1,1,1,1,1,1], [1,2,3,4,5,6,7,8,9,10], [0,0,0,0,0,0,0,0,0,0]], type: :f64)
    Nx.iota({@size})
    |> Nx.add(1)
    |> Nx.multiply(Nx.pow(genomes, 2))
    |> Nx.sum(axes: [1])
    |> Nx.negate
  end
end
