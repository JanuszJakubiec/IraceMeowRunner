defmodule MeowRunnerTest do
  use ExUnit.Case
  doctest MeowRunner

  test "greets the world" do
    assert MeowRunner.hello() == :world
  end
end
