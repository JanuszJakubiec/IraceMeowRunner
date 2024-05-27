defmodule XinShe do
  import Nx.Defn
  import Tuple

  @size 10

  @two_pi 2 * :math.pi()

  defn evaluate(genomes) do
    #genomes = Nx.tensor([[1,1,1,1,1,1,1,1,1,1], [1,2,3,4,5,6,7,8,9,10], [0,0,0,0,0,0,0,0,0,0]], type: :f64)

    {head, _} = Nx.shape(genomes)

    sums =
      10000 * (1+ (Nx.exp(-1 * Nx.sum(Nx.pow(genomes/15,10))) - 2* Nx.exp(-1 * Nx.sum(Nx.pow(genomes, 2))) ) * Nx.product(Nx.pow(Nx.cos(genomes),2) ))


    -sums
  end

end
