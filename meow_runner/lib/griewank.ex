defmodule Griewank do
  import Nx.Defn

  @size 100
  @denominator 4000

  defn evaluate(genomes) do
    #genomes = Nx.tensor([[1,1,1,1,1,1,1,1,1,1], [1,2,3,4,5,6,7,8,9,10], [0,0,0,0,0,0,0,0,0,0]], type: :f64)

    first = genomes
    |> Nx.pow(2)
    |> Nx.divide(@denominator)
    |> Nx.sum(axes: [1])

    second = genomes
    |> Nx.divide(Nx.iota({@size}) |> Nx.add(1) |> Nx.sqrt)
    |> Nx.cos
    |> Nx.product(axes: [1])
    |> Nx.negate

    first
    |> Nx.add(second)
    |> Nx.add(1)
    |> Nx.negate
  end
end
