defmodule MeowRunner.ApplicationCopy do
  def start() do
    IO.puts("Starting the app\n")
    Nx.Defn.global_default_options(compiler: EXLA)
    Node.start(:"meow_runner@127.0.0.1")
  end

  def run_rastrigin(number_of_evaluations, populations_n, genomes_modification_params, emigration_params) do
    algorithm =
      Meow.objective(
        &Rastrigin.evaluate_rastrigin/1
      )
      |> Meow.add_pipeline(
        MeowNx.Ops.init_real_random_uniform(100, 10, -5.12, 5.12),
        Meow.pipeline(
          genomes_modification_settings(genomes_modification_params) ++
          emigration_settings(emigration_params, populations_n) ++
          [MeowNx.Ops.log_best_individual,
          MeowNx.Ops.log_metrics(
            %{
              fitness_max: &MeowNx.Metric.fitness_max/2
            }
          ),
          Meow.Ops.max_generations(div(number_of_evaluations, populations_n))
        ]),
        duplicate: populations_n
      )
    result = Meow.run(algorithm)
    population_reports = result.population_reports
    List.foldl(population_reports,
               :no_value,
               fn (elem, acc) ->
                 value = elem.population.log.best_individual.fitness
                 case acc do
                   :no_value -> value
                   acc_value when acc_value < value -> value
                   _ -> acc
                 end
               end)
  end

  def genomes_modification_settings({false, mutation_selection, crossover_params, mutation_params}) do
    [selection_settings({mutation_selection, 1.0})] ++
    crossover_settings(crossover_params) ++
    mutation_settings(mutation_params)
  end
  def genomes_modification_settings({true, fittest_survival, fittest_selection, mutation_selection, crossover_params, mutation_params}) do
    fittest_survival = 2 * fittest_survival
    [Meow.Ops.split_join([
      Meow.pipeline([selection_settings({fittest_selection, fittest_survival})]),
      Meow.pipeline(
        [selection_settings({mutation_selection, 100 - fittest_survival})] ++
        crossover_settings(crossover_params) ++
        mutation_settings(mutation_params)
      )
    ])]
  end

  def emigration_settings({false, _, _, _, _, _}, _) do
    []
  end
  def emigration_settings(_, 1) do
    []
  end
  def emigration_settings({_, interval, topology, emigration_size, emigrate_selection, imigrate_selection}, _) do
    [Meow.Ops.emigrate(selection_settings({emigrate_selection, emigration_size}), topology_settings(topology), interval: interval),
    Meow.Ops.immigrate(imigration_selection_settings(imigrate_selection), interval: interval)]
  end

  def topology_settings(:ring) do
    &Meow.Topology.ring/2
  end
  def topology_settings(:star) do
    &Meow.Topology.star/2
  end
  def topology_settings(:mesh2d) do
    &Meow.Topology.mesh2d/2
  end
  def topology_settings(:mesh3d) do
    &Meow.Topology.mesh3d/2
  end
  def topology_settings(:fully_connected) do
    &Meow.Topology.fully_connected/2
  end

  def mutation_settings({:shift_gaussian, probability, sigma}) do
    [MeowNx.Ops.mutation_shift_gaussian(probability, sigma: sigma)]
  end
  def mutation_settings({:replace_uniform, probability, _}) do
    [MeowNx.Ops.mutation_replace_uniform(probability, -5.12, 5.12)]
  end
  def mutation_settings({:bit_flip, probability, _}) do
    [MeowNx.Ops.mutation_bit_flip(probability)]
  end

  def crossover_settings({:uniform, probability}) do
    [MeowNx.Ops.crossover_uniform(probability)]
  end
  def crossover_settings({:blend_alpha, alpha}) do
    [MeowNx.Ops.crossover_blend_alpha(alpha)]
  end
  def crossover_settings({:multi_point, points}) do
    [MeowNx.Ops.crossover_multi_point(points)]
  end

  def selection_settings({:natural, n_selected}) do
    MeowNx.Ops.selection_natural(n_selected)
  end
  def selection_settings({:tournament, n_selected}) do
    MeowNx.Ops.selection_tournament(n_selected)
  end
  def selection_settings({:roulette, n_selected}) do
    MeowNx.Ops.selection_roulette(n_selected)
  end
  def selection_settings({:sus, n_selected}) do
    MeowNx.Ops.selection_stochastic_universal_sampling(n_selected)
  end

  def imigration_selection_settings(:natural) do
    &MeowNx.Ops.selection_natural(&1)
  end
  def imigration_selection_settings(:tournament) do
    &MeowNx.Ops.selection_tournament(&1)
  end
  def imigration_selection_settings(:roulette) do
    &MeowNx.Ops.selection_roulette(&1)
  end
  def imigration_selection_settings(:sus) do
    &MeowNx.Ops.selection_stochastic_universal_sampling(&1)
  end
end
