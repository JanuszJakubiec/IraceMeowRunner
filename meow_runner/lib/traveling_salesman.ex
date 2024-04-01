defmodule TravelingSalesman do
  import Nx.Defn

  defn crossover(parents) do
    # crossover function produces new genomes based on given parents genomes.
    # Given genomes are arranged into pairs and new offspring is produced.

    # parents = parents |> Nx.Defn.Kernel.print_value(limit: :infinity)
    # parents =
    #  Nx.tensor([
    #    [2, 3, 7, 1, 6, 0, 5, 4],
    #    [3, 1, 4, 0, 5, 7, 2, 6],
    #    [3, 4, 0, 2, 7, 1, 6, 5],
    #    [4, 2, 5, 1, 6, 0, 3, 7],
    #    [0, 1, 2, 3, 4, 5, 6, 7],
    #    [7, 6, 5, 4, 3, 2, 1, 0]
    #  ])

    {genomes_n, length} = Nx.shape(parents)
    half_n = div(genomes_n, 2)

    # Getting genomes split points
    split_idx = generate_random_points(half_n, length)

    # Swapping adjacent parents rows, beginning order: 0, 1, 2, 3
    # order after swap: 1, 0, 3, 2
    swapped_parents = swap_adjacent_rows(parents)

    # Swapping desired genomes fragments between parents. The value equal to the
    # length of genomes is inserted in places where the values are not swapped
    swapped_genomes_fragments =
      Nx.less_equal(split_idx, Nx.iota({1, 1, length}))
      |> Nx.sum(axes: [1])
      |> Nx.remainder(2)
      |> duplicate_rows()
      |> Nx.select(swapped_parents, length)
      # After swapping, the genomes are expanded
      # genomes swapped looking like this:
      # [[3, 2, 1],
      #  [3, 1, 2]]
      # Are changed into:
      # [[[3, 3, 3], [2, 2, 2], [1, 1, 1]],
      #  [[3, 3, 3], [1, 1, 1], [2, 2, 2]]]
      |> Nx.reshape({genomes_n, length, 1})
      |> Nx.broadcast({genomes_n, length, length})

    # Getting starting and ending indices of swapped fragments
    {start_indices, end_indices, _} = get_lower_upper_length_from_split_points(split_idx)

    # Expanding ending indices
    start_indices_expanded =
      start_indices
      |> Nx.reshape({half_n, 1})
      |> duplicate_rows()
      |> Nx.broadcast({genomes_n, length})
      |> Nx.reshape({genomes_n, length, 1})

    # Expanding starting indices
    end_indices_expanded =
      end_indices
      |> Nx.reshape({half_n, 1})
      |> duplicate_rows()
      |> Nx.broadcast({genomes_n, length})
      |> Nx.reshape({genomes_n, length, 1})

    # Calculate the effective indices for the original genome positions
    effective_indices =
      Nx.iota({genomes_n, length, 1}, axis: 1)
      |> Nx.add(end_indices_expanded)
      |> Nx.remainder(length)

    # Effective indices with added parent indice at the beginning
    indices =
      Nx.concatenate([Nx.iota({genomes_n, length, 1}, axis: 0), effective_indices], axis: 2)

    # Expanding original parents to make it possible to determine the positions
    # of swapped elements in their original genomes
    expaned_original_parents =
      parents
      |> Nx.reshape({genomes_n, length, 1})
      |> Nx.broadcast({genomes_n, length, length})
      |> Nx.transpose(axes: [0, 2, 1])

    # Identify and move the non-crossover fragments
    # By comparing the swapped_genomes_fragments and expaned_original_parents
    # I identify the location of swapped elements in original parents
    non_crossover_move =
      Nx.equal(swapped_genomes_fragments, expaned_original_parents)
      |> Nx.transpose(axes: [0, 2, 1])
      # Summing is used to create a tensor for each parent eg. [0, 1, 0, 1, 0],
      # That identifies the location of swapped elements in original genomes
      |> Nx.sum(axes: [2])
      |> Nx.gather(indices)
      |> Nx.select(
        Nx.broadcast(length, {genomes_n, length}),
        Nx.iota({genomes_n, length}, axis: 1)
      )
      |> Nx.sort(axis: 1)
      |> Nx.reshape({genomes_n, length, 1})
      |> Nx.add(end_indices_expanded)
      |> Nx.remainder(length)

    adjustment_factor =
      Nx.iota({genomes_n, length, 1}, axis: 0)
      |> Nx.remainder(2)
      |> Nx.select(-1, 1)

    # Indices of swapped fragments of genomes
    indices_swapped =
      Nx.concatenate(
        [Nx.iota({genomes_n, length, 1}, axis: 0) + adjustment_factor, effective_indices],
        axis: 2
      )

    # Indices of not swapped fragments of genomes
    indices_not_swapped =
      Nx.concatenate([Nx.iota({genomes_n, length, 1}, axis: 0), non_crossover_move], axis: 2)

    pred =
      effective_indices >= start_indices_expanded and
        effective_indices <
          end_indices_expanded
          |> Nx.broadcast({genomes_n, length, 2})

    crossover_indices = Nx.select(pred, indices_swapped, indices_not_swapped)

    # Final creation of crossovered genomes
    Nx.gather(parents, crossover_indices)
  end

  defn mutation(genomes, opts \\ []) do
    opts = keyword!(opts, [:probability])
    probability = opts[:probability]

    # genomes =
    #  Nx.tensor([
    #    [2, 3, 7, 1, 6, 0, 5, 4],
    #    [3, 1, 4, 0, 5, 7, 2, 6],
    #    [3, 4, 0, 2, 7, 1, 6, 5],
    #    [4, 2, 5, 1, 6, 0, 3, 7],
    #    [0, 1, 2, 3, 4, 5, 6, 7],
    #    [7, 6, 5, 4, 3, 2, 1, 0]
    #  ])

    {genomes_n, length} = Nx.shape(genomes)

    # Tensor determining if the given tensor is mutating
    mutate? =
      Nx.random_uniform({genomes_n, 1})
      |> Nx.less(probability)
      |> Nx.broadcast({genomes_n, length})

    # Getting mutation first indice, second indice and distance beteween elements that will be swapped
    {lower, upper, generated_length} =
      generate_random_points(genomes_n, length)
      |> get_lower_upper_length_from_split_points

    # Distance between vertices that will be swapped
    generated_length =
      generated_length
      |> Nx.reshape({genomes_n, 1})
      |> Nx.broadcast({genomes_n, length})

    # first indice, that will be added
    lower_add =
      Nx.iota({genomes_n, length}, axis: 1)
      |> Nx.add(generated_length)

    # second indice, that will be subtracted
    upper_subtr =
      Nx.iota({genomes_n, length}, axis: 1)
      |> Nx.subtract(generated_length)

    lower_pred = Nx.take(Nx.eye(length), lower)
    upper_pred = Nx.take(Nx.eye(length), upper)

    idx = Nx.iota({genomes_n, length}, axis: 1)
    idx = Nx.select(lower_pred, lower_add, idx)

    idx =
      Nx.select(upper_pred, upper_subtr, idx)
      |> Nx.reshape({genomes_n, length, 1})

    idx = Nx.concatenate([Nx.iota({genomes_n, length, 1}, axis: 0), idx], axis: 2)
    mutated_genomes = Nx.gather(genomes, idx)

    Nx.select(mutate?, mutated_genomes, genomes)
  end

  defn init_function(opts \\ []) do
    opts = keyword!(opts, [:genomes_n, :length])
    genomes_n = opts[:genomes_n]
    genome_length = opts[:length]

    shuffled =
      Nx.random_uniform({genomes_n, genome_length})
      |> Nx.argsort(axis: 1)
  end

  defn evaluate(genomes) do
    # genomes = Nx.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 7, 6, 5, 4]])

    edges =
      Nx.tensor([
        [0, 1, 2, 2, 2, 2, 2, 2],
        [2, 0, 1, 2, 2, 2, 2, 2],
        [2, 2, 0, 1, 2, 2, 2, 2],
        [2, 2, 2, 0, 1, 2, 2, 2],
        [2, 2, 2, 2, 0, 1, 2, 2],
        [2, 2, 2, 2, 2, 0, 1, 2],
        [2, 2, 2, 2, 2, 2, 0, 1],
        [1, 2, 2, 2, 2, 2, 2, 0]
      ])

    {genomes_n, length} = Nx.shape(genomes)
    slice = Nx.slice(genomes, [0, 0], [genomes_n, 1])

    # Duplicating the vertices
    transformed_genomes =
      Nx.reshape(genomes, {genomes_n, length, 1})
      |> Nx.tile([2])
      |> Nx.reshape({genomes_n, length * 2})

    # Transforming the genomes to edges. Adding the edge from last vertice to the first one
    transformed_genomes =
      transformed_genomes
      |> Nx.put_slice(
        [0, 0],
        Nx.slice(transformed_genomes, [0, 1], [genomes_n, length * 2 - 1])
      )
      |> Nx.put_slice([0, length * 2 - 1], slice)
      |> Nx.reshape({genomes_n, length, 2})

    # Gathering the edges from the edges tensor, evaluating each genome
    -Nx.gather(edges, transformed_genomes)
    |> Nx.sum(axes: [1])
  end

  defn get_lower_upper_length_from_split_points(idx) do
    flatten_idx =
      Nx.flatten(idx, axes: [1, 2])
      |> Nx.transpose(axes: [1, 0])

    greater_idx = Nx.select(flatten_idx[0] > flatten_idx[1], flatten_idx[0], flatten_idx[1])
    lesser_idx = Nx.select(flatten_idx[0] < flatten_idx[1], flatten_idx[0], flatten_idx[1])
    crossover_length = greater_idx - lesser_idx
    {lesser_idx, greater_idx, crossover_length}
  end

  defn generate_random_points(elements_n, genome_length) do
    split_points = 2

    random_idx_without_replacement(
      shape: {elements_n, split_points, 1},
      min: 0,
      max: genome_length,
      axis: 1
    )
  end

  defn swap_adjacent_rows(t) do
    {n, m} = Nx.shape(t)
    half_n = div(n, 2)

    t
    |> Nx.reshape({half_n, 2, m})
    |> Nx.reverse(axes: [1])
    |> Nx.reshape({n, m}) end

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
