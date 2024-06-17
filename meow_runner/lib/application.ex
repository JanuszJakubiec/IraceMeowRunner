defmodule MeowRunner.Application do
  use Application

  def start(_type, _args) do
    IO.puts("Starting the app\n")
    Nx.Defn.global_default_options(compiler: EXLA)
  end

  def run_meow(
        function,
        half_population_size,
        number_of_evaluations,
        populations_n,
        genomes_modification_params,
        emigration_params
      ) do
    algorithm =
      Meow.objective(get_function(function))
      |> Meow.add_pipeline(
        init_function(function_type(function), half_population_size, 16),
        Meow.pipeline(
          genomes_modification_settings(half_population_size * 2, genomes_modification_params) ++
            emigration_settings(emigration_params, populations_n) ++
            [
              MeowNx.Ops.log_best_individual(),
              MeowNx.Ops.log_metrics(%{
                fitness_max: &MeowNx.Metric.fitness_max/2
              }),
              Meow.Ops.max_generations(div(number_of_evaluations, populations_n))
            ]
        ),
        duplicate: populations_n
      )

    result = Meow.run(algorithm)
    population_reports = result.population_reports

    List.foldl(
      population_reports,
      :no_value,
      fn elem, acc ->
        IO.inspect(elem.population.log.best_individual, limit: :infinity)
        value = elem.population.log.best_individual.fitness

        case acc do
          :no_value -> value
          acc_value when acc_value < value -> value
          _ -> acc
        end
      end
    )
  end

  defp get_function(:backpack), do: &Backpack.evaluate/1
  defp get_function(:rastrigin), do: &Rastrigin.evaluate/1
  defp get_function(:sumSquares), do: &SumSquares.evaluate/1
  defp get_function(:ackley), do: &Ackley.evaluate/1
  defp get_function(:griewank), do: &Griewank.evaluate/1
  defp get_function(:myFunction), do: &MyFunction.evaluate/1
  defp get_function(:max_cut), do: &MaxCut.evaluate/1
  defp get_function(:n_queen), do: &NQueen.evaluate/1
  defp get_function(:traveling_salesman), do: &TravelingSalesman.evaluate/1

  defp function_type(:backpack), do: :discrete
  defp function_type(:rastrigin), do: :continous
  defp function_type(:sumSquares), do: :continous
  defp function_type(:ackley), do: :continous
  defp function_type(:griewank), do: :continous
  defp function_type(:myFunction), do: :continous
  defp function_type(:max_cut), do: :discrete
  defp function_type(:n_queen), do: :discrete
  defp function_type(:traveling_salesman), do: :traveling_salesman

  defp init_function(:discrete, half_population_size, size) do
    MeowNx.Ops.init_binary_random_uniform(half_population_size * 2, size)
  end

  defp init_function(:continous, half_population_size, size) do
    MeowNx.Ops.init_real_random_uniform(half_population_size * 2, size, -30.12, 30.12)
  end

  defp init_function(:traveling_salesman, half_population_size, size) do
    opts = [genomes_n: half_population_size * 2, length: size]

    %Meow.Op{
      name: "[Nx] Initialization: initialization for traveling salesman",
      requires_fitness: false,
      invalidates_fitness: true,
      in_representations: :any,
      out_representation: MeowNx.permutation_representation(),
      impl: fn population, _ctx ->
        Meow.Population.map_genomes(population, fn _genomes ->
          TravelingSalesman.init_function(opts)
        end)
      end
    }
  end

  defp genomes_modification_settings(
         _,
         {false, mutation_selection, crossover_params, mutation_params}
       ) do
    [selection_settings({mutation_selection, 1.0})] ++
      crossover_settings(crossover_params) ++
      mutation_settings(mutation_params)
  end

  defp genomes_modification_settings(
         genomes_n,
         {true, fittest_survival, fittest_selection, mutation_selection, crossover_params,
          mutation_params}
       ) do
    fittest_survival = 2 * fittest_survival

    [
      Meow.Ops.split_join([
        Meow.pipeline([selection_settings({fittest_selection, fittest_survival})]),
        Meow.pipeline(
          [selection_settings({mutation_selection, genomes_n - fittest_survival})] ++
            crossover_settings(crossover_params) ++
            mutation_settings(mutation_params)
        )
      ])
    ]
  end

  defp emigration_settings({false, _, _, _, _, _}, _) do
    []
  end

  defp emigration_settings(_, 1) do
    []
  end

  defp emigration_settings(
         {_, migration_interval, topology, emigration_size, emigrate_selection,
          imigrate_selection},
         _
       ) do
    [
      Meow.Ops.emigrate(
        selection_settings({emigrate_selection, emigration_size}),
        topology_settings(topology),
        interval: migration_interval,
        number_of_targets: :all
      ),
      imigration_selection_settings(imigrate_selection, migration_interval)
    ]
  end

  defp topology_settings(:ring) do
    &Meow.Topology.ring/2
  end

  defp topology_settings(:star) do
    &Meow.Topology.star/2
  end

  defp topology_settings(:mesh2d) do
    &Meow.Topology.mesh2d/2
  end

  defp topology_settings(:mesh3d) do
    &Meow.Topology.mesh3d/2
  end

  defp topology_settings(:fully_connected) do
    &Meow.Topology.fully_connected/2
  end

  defp mutation_settings({:traveling_salesman, probability, _}) do
    opts = [probability: probability]

    [
      %Meow.Op{
        name: "[Nx] Mutation: mutation for traveling salesman",
        requires_fitness: false,
        invalidates_fitness: true,
        in_representations: [MeowNx.permutation_representation()],
        impl: fn population, _ctx ->
          Meow.Population.map_genomes(population, fn genomes ->
            TravelingSalesman.mutation(genomes, opts)
          end)
        end
      }
    ]
  end

  defp mutation_settings({:shift_gaussian, probability, sigma}) do
    [MeowNx.Ops.mutation_shift_gaussian(probability, sigma: sigma)]
  end

  defp mutation_settings({:replace_uniform, probability, _}) do
    [MeowNx.Ops.mutation_replace_uniform(probability, -5.12, 5.12)]
  end

  defp mutation_settings({:bit_flip, probability, _}) do
    [MeowNx.Ops.mutation_bit_flip(probability)]
  end

  defp crossover_settings({:traveling_salesman, _}) do
    [
      %Meow.Op{
        name: "[Nx] Crossover: crossover for traveling salesman",
        requires_fitness: false,
        invalidates_fitness: true,
        in_representations: [MeowNx.permutation_representation()],
        impl: fn population, _ctx ->
          Meow.Population.map_genomes(population, fn genomes ->
            TravelingSalesman.crossover(genomes)
          end)
        end
      }
    ]
  end

  defp crossover_settings({:uniform, probability}) do
    [MeowNx.Ops.crossover_uniform(probability)]
  end

  defp crossover_settings({:blend_alpha, alpha}) do
    [MeowNx.Ops.crossover_blend_alpha(alpha)]
  end

  defp crossover_settings({:multi_point, points}) do
    [MeowNx.Ops.crossover_multi_point(points)]
  end

  defp selection_settings({:natural, n_selected}) do
    MeowNx.Ops.selection_natural(n_selected)
  end

  defp selection_settings({:tournament, n_selected}) do
    MeowNx.Ops.selection_tournament(n_selected)
  end

  defp selection_settings({:roulette, n_selected}) do
    MeowNx.Ops.selection_roulette(n_selected)
  end

  defp selection_settings({:sus, n_selected}) do
    MeowNx.Ops.selection_stochastic_universal_sampling(n_selected)
  end

  defp imigration_selection_settings(:natural, migration_interval) do
    Meow.Ops.immigrate(&MeowNx.Ops.selection_natural(&1), interval: migration_interval)
  end

  defp imigration_selection_settings(:tournament, migration_interval) do
    Meow.Ops.immigrate(&MeowNx.Ops.selection_tournament(&1), interval: migration_interval)
  end

  defp imigration_selection_settings(:roulette, migration_interval) do
    Meow.Ops.immigrate(&MeowNx.Ops.selection_roulette(&1), interval: migration_interval)
  end

  defp imigration_selection_settings(:sus, migration_interval) do
    Meow.Ops.immigrate(&MeowNx.Ops.selection_stochastic_universal_sampling(&1),
      interval: migration_interval
    )
  end
end
