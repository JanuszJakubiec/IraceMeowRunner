defmodule NQueen do
  import Nx.Defn

  defn evaluate(_genomes) do
    size = 8

    genomes =
      Nx.tensor([
        [
          [1, 0, 0, 0, 0, 0, 1, 1],
          [0, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 0, 0],
          [1, 0, 0, 0, 0, 0, 1, 0],
          [1, 1, 0, 0, 0, 0, 0, 1]
        ],
        [
          [1, 0, 0, 0, 0, 0, 1, 1],
          [0, 1, 0, 0, 0, 0, 0, 1],
          [0, 1, 0, 0, 0, 1, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [1, 0, 0, 0, 0, 0, 1, 0]
        ]
      ])

    genomes = genomes |> Nx.flatten(axes: [1, 2])

    {genomes_n, _} = Nx.shape(genomes)

    genomes = genomes |> Nx.reshape({genomes_n, size, size})

    Nx.sum(genomes, axes: [1, 2])
    |> Nx.subtract(count_columns(genomes))
    |> Nx.subtract(count_rows(genomes))
    |> Nx.subtract(count_diagonal(genomes))
    |> Nx.subtract(count_diagonal(Nx.reverse(genomes, axes: [2])))
  end

  defn count_columns(genomes) do
    Nx.sum(genomes, axes: [1])
    |> Nx.subtract(1)
    |> Nx.max(0)
    |> Nx.sum(axes: [1])
  end

  defn count_rows(genomes) do
    Nx.sum(genomes, axes: [2])
    |> Nx.subtract(1)
    |> Nx.max(0)
    |> Nx.sum(axes: [1])
  end

  defn count_diagonal(genomes) do
    {genomes_n, size, _} = Nx.shape(genomes)

    acc =
      sum_genomes_diagonal(genomes, Nx.tensor(-size + 1))
      |> Nx.broadcast({2 * size - 1, genomes_n})

    {_, acc} =
      while {x = -size + 2, acc}, Nx.less(x, size) do
        sum =
          sum_genomes_diagonal(genomes, x)
          |> Nx.broadcast({2 * size - 1, genomes_n})

        pred =
          Nx.iota({2 * size - 1, 2}, axis: 0)

        {x + 1, Nx.select(pred == x + size - 1, sum, acc)}
      end

    Nx.sum(acc, axes: [0])
  end

  defn sum_genomes_diagonal(genomes, offset) do
    {genomes_n, data_size, _} = Nx.shape(genomes)
    first_indice = Nx.iota({data_size, 1})
    second_indice = Nx.iota({data_size, 1}) + offset

    genomes_with_added_element =
      Nx.concatenate([genomes, Nx.broadcast(0, {1, data_size, data_size})], axis: 0)

    pred =
      check_if_indices_are_proper(second_indice, data_size)
      |> Nx.broadcast({genomes_n, data_size, 3})

    indices =
      Nx.concatenate([first_indice, second_indice], axis: 1)
      |> Nx.broadcast({genomes_n, data_size, 2})

    indices = Nx.concatenate([Nx.iota({genomes_n, data_size, 1}, axis: 0), indices], axis: 2)

    selected_indices =
      Nx.select(
        pred,
        indices,
        Nx.broadcast(Nx.tensor([genomes_n, 0, 0]), {genomes_n, data_size, 3})
      )

    Nx.gather(genomes_with_added_element, selected_indices)
    |> Nx.sum(axes: [1])
    |> Nx.subtract(1)
    |> Nx.max(0)
  end

  defn check_if_indices_are_proper(y_indice, size) do
    y_indice >= 0 and y_indice < size
  end
end
