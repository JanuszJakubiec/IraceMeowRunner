defmodule MyFunction do

  @border_value 7

  def evaluate(genomes) do
    #genomes = Nx.tensor([[1,1,1,1,1,1,1,1,1,1], [1,2,3,4,5,6,7,8,9,10], [0,0,0,0,0,0,0,0,0,0]], type: :f64)
    norm = genomes
    |> Nx.pow(2)
    |> Nx.sum(axes: [1])

    Nx.select(Nx.greater_equal(@border_value, norm), 0, @border_value)
    |> Nx.negate
  end

end
