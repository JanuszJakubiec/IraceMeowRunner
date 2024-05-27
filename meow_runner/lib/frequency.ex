defmodule Frequency do
  import Nx.Defn

  @size 100

  @two_pi 2 * :math.pi()

  defn evaluate(genomes) do
    #genomes = Nx.tensor([[1,1,1,1,1,1,1,1,1,1], [1,2,3,4,5,6,7,8,9,10], [0,0,0,0,0,0,0,0,0,0]], type: :f64)
    sums =
      #(@size + Nx.pow(genomes, 2) - @size * Nx.cos(genomes * @two_pi))



      (Nx.abs((genomes * Nx.sin(genomes) + 0.1 * genomes)))
      #|> Nx.abs()
      |> Nx.sum(axes: [1])

    -sums
  end
end
