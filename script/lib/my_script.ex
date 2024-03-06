defmodule MyScript do
  def main(args \\ []) do
    args_map = parse_args(args)
    Node.start(:"irace_elixir@127.0.0.1")
    {number_of_evaluations, _} = Integer.parse(args_map["number_of_evaluations"])
    {populations_n, _} = Integer.parse(args_map["populations"])
    {half_population_size, _} = Integer.parse(args_map["half_population_size"])
    problem = parse_atom(args_map["problem"])
    genomes_modification_params = get_genomes_modification_params(args_map)
    emigration_params = get_emigration_settings(args_map, populations_n)

    res =
      :rpc.call(:"meow_runner@127.0.0.1", MeowRunner.Application, :run_meow, [
        problem,
        half_population_size,
        number_of_evaluations,
        populations_n,
        genomes_modification_params,
        emigration_params
      ])

    IO.puts(res * -1)
  end

  def get_emigration_settings(%{"migrate_animals" => "false"}, _) do
    {false, :null, :null, :null, :null, :null}
  end

  def get_emigration_settings(_, 1) do
    {false, :null, :null, :null, :null, :null}
  end

  def get_emigration_settings(args_map, _) do
    {true, parse_emigration_interval(args_map), parse_topology(args_map),
     parse_emigration_size(args_map), parse_emigration_selection(args_map),
     parse_imigration_selection(args_map)}
  end

  def parse_emigration_interval(%{"interval" => interval}) do
    parse_integer(interval)
  end

  def parse_topology(%{"topology" => topology}) do
    parse_atom(topology)
  end

  def parse_emigration_size(%{"emigration_size" => size}) do
    parse_integer(size)
  end

  def parse_emigration_selection(%{"emigration_selection" => selection}) do
    parse_atom(selection)
  end

  def parse_imigration_selection(%{"imigration_selection" => selection}) do
    parse_atom(selection)
  end

  def get_genomes_modification_params(%{"preserve_best_genes" => "true"} = args_map) do
    {true, get_fittest_survival_number(args_map), get_fittest_selection(args_map),
     get_mutation_selection(args_map), get_crossover_params(args_map),
     get_mutation_params(args_map)}
  end

  def get_genomes_modification_params(%{"preserve_best_genes" => "false"} = args_map) do
    {false, get_mutation_selection(args_map), get_crossover_params(args_map),
     get_mutation_params(args_map)}
  end

  def get_fittest_survival_number(%{"fittest_survival" => fittest_survival}) do
    parse_integer(fittest_survival)
  end

  def get_fittest_selection(%{"fittest_genomes_selection" => selection}) do
    parse_atom(selection)
  end

  def get_mutation_selection(%{"mutation_genomes_selection" => selection}) do
    parse_atom(selection)
  end

  def get_crossover_params(%{"crossover" => "uniform"} = args_map) do
    {:uniform, parse_float(args_map["crossover_probability"])}
  end

  def get_crossover_params(%{"crossover" => "blend_alpha"} = args_map) do
    {:blend_alpha, parse_float(args_map["alpha"])}
  end

  def get_crossover_params(%{"crossover" => "multi_point"} = args_map) do
    {:uniform, parse_integer(args_map["points"])}
  end

  def get_mutation_params(%{"mutation" => "shift_gaussian"} = args_map) do
    {:shift_gaussian, parse_float(args_map["mutation_probability"]),
     parse_float(args_map["mutation_sigma"])}
  end

  def get_mutation_params(%{"mutation" => "replace_uniform"} = args_map) do
    {:replace_uniform, parse_float(args_map["mutation_probability"]), :null}
  end

  def get_mutation_params(%{"mutation" => "bit_flip"} = args_map) do
    {:bit_flip, parse_float(args_map["mutation_probability"]), :null}
  end

  def parse_atom(atom_to_parse) do
    String.to_atom(atom_to_parse)
  end

  def parse_float(float_to_parse) do
    {float, _} = Float.parse(float_to_parse)
    float
  end

  def parse_integer(integer_to_parse) do
    {integer, _} = Integer.parse(integer_to_parse)
    integer
  end

  @spec parse_args(list) :: any
  def parse_args(args) do
    List.foldl(args, %{}, fn elem, acc -> parse_arg(elem, acc) end)
  end

  def parse_arg(arg, acc) do
    case String.starts_with?(arg, "--") do
      true ->
        {_, parsed_arg} = String.split_at(arg, 2)

        case String.split(parsed_arg, "=", trim: true) do
          [key, value] ->
            Map.put(acc, key, value)

          _ ->
            acc
        end

      false ->
        acc
    end
  end
end
