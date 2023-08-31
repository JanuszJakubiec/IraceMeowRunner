defmodule Ackley do
  import Nx.Defn

  @size 10
  @a 20
  @b 0.2
  @c 2 * :math.pi()

  defn evaluate_ackley(genomes) do
    genomes = Nx.tensor([[1,1,1,1,1,1,1,1,1,1], [1,2,3,4,5,6,7,8,9,10], [0,0,0,0,0,0,0,0,0,0]], type: :f64)
    first_part = genomes
    |> Nx.pow(2)
    |> Nx.sum(axes: [1])
    |> Nx.divide(@size)
    |> Nx.sqrt
    |> Nx.multiply(-@b)
    |> Nx.exp
    |> Nx.multiply(-@a)

    second_part = genomes
    |> Nx.multiply(@c)
    |> Nx.cos
    |> Nx.sum(axes: [1])
    |> Nx.divide(@size)
    |> Nx.exp
    |> Nx.negate

    first_part
    |> Nx.add(second_part)
    |> Nx.add(@a)
    |> Nx.add(Nx.exp(1))
  end
end
