defmodule TravelingSalesman do
  import Nx.Defn

  defn crossover(_parents) do
    parents =
      Nx.tensor([
        [2, 3, 7, 1, 6, 0, 5, 4],
        [3, 1, 4, 0, 5, 7, 2, 6],
        [0, 1, 2, 3, 5, 4, 6, 7],
        [0, 1, 2, 3, 7, 6, 5, 4]
      ])
      |> Nx.Defn.Kernel.print_value()

    {n, length} = Nx.shape(parents)
    points = 2
    half_n = div(n, 2)

    swapped_parents = swap_adjacent_rows(parents)

    split_idx =
      random_idx_without_replacement(
        shape: {half_n, points, 1},
        min: 1,
        max: length,
        axis: 1
      )

    swap? =
      Nx.less_equal(split_idx, Nx.iota({1, 1, length}))
      |> Nx.sum(axes: [1])
      |> Nx.remainder(2)
      |> duplicate_rows()
      |> Nx.Defn.Kernel.print_value()

    swapped_parents =
      Nx.select(swap?, swapped_parents, length)

    length_plus_one = transform(length, &(&1 + 1))

    swapped_parents_and_added_parents =
      Nx.take(Nx.eye(length_plus_one), swapped_parents)
      |> Nx.transpose(axes: [0, 2, 1])
      |> Nx.sum(axes: [2])

    # |> Nx.add(Nx.iota({length_plus_one}) == length)
    # |> Nx.take_along_axis(parents, axis: 1)
    # |> Nx.add(swap?)
    # |> Nx.select(swapped_parents, parents)

    swapped_parents_and_added_parents
  end

  defn mutation(genomes) do
  end

  defn generate_init(genomes_n, genome_length) do
    key = Nx.Random.key(42)
    genomes = Nx.iota({genomes_n, genome_length}, axis: 1)
    {shuffled, _} = Nx.Random.shuffle(key, genomes, axis: 1, independent: true)
    shuffled
  end

  defn evaluation(_genomes) do
    genomes = Nx.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 7, 6, 5, 4]])

    edges =
      Nx.tensor([
        [0, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 2, 1, 1, 1, 1, 1],
        [1, 1, 0, 3, 1, 1, 1, 1],
        [1, 1, 1, 0, 4, 1, 1, 1],
        [1, 1, 1, 1, 0, 5, 1, 1],
        [1, 1, 1, 1, 1, 0, 6, 1],
        [1, 1, 1, 1, 1, 1, 0, 7],
        [8, 1, 1, 1, 1, 1, 1, 0]
      ])

    {genomes_n, genomes_l} = Nx.shape(genomes)
    slice = Nx.slice(genomes, [0, 0], [genomes_n, 1])

    transformed_genomes =
      Nx.reshape(genomes, {genomes_n, genomes_l, 1})
      |> Nx.tile([2])
      |> Nx.reshape({genomes_n, transform(genomes_l, &(&1 * 2))})

    transformed_genomes =
      transformed_genomes
      |> Nx.put_slice(
        [0, 0],
        Nx.slice(transformed_genomes, [0, 1], [genomes_n, transform(genomes_l, &(&1 * 2 - 1))])
      )
      |> Nx.put_slice([0, transform(genomes_l, &(&1 * 2 - 1))], slice)
      |> Nx.reshape({genomes_n, genomes_l, 2})

    Nx.gather(edges, transformed_genomes)
    |> Nx.sum(axes: [1])
  end

  defn swap_adjacent_rows(t) do
    {n, m} = Nx.shape(t)
    half_n = div(n, 2)

    t
    |> Nx.reshape({half_n, 2, m})
    |> Nx.reverse(axes: [1])
    |> Nx.reshape({n, m})
  end

  defn random_idx_without_replacement(opts \\ []) do
    opts = keyword!(opts, [:shape, :min, :max, :axis])
    shape = opts[:shape]
    min = opts[:min]
    max = opts[:max]
    axis = opts[:axis]

    range = max - min

    sample_size = transform(shape, &elem(&1, axis))
    random_shape = transform(shape, &put_elem(&1, axis, range))

    random_shape
    |> Nx.random_uniform()
    |> Nx.argsort(axis: axis)
    |> Nx.slice_along_axis(0, sample_size, axis: axis)
    |> Nx.add(min)
  end

  defn duplicate_rows(t) do
    {n, m} = Nx.shape(t)
    twice_n = transform(n, &(&1 * 2))

    t
    |> Nx.tile([1, 2])
    |> Nx.reshape({twice_n, m})
  end
end
