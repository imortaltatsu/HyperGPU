-module(gpu_test).
-export([run_tests/0]).

run_tests() ->
    io:format("Starting GPU device tests...~n"),
    
    % Initialize device
    {ok, State1} = dev_gpu_nif:init(#{}, #{}, #{}),
    io:format("Device initialized successfully~n"),
    
    % Test 1: Simple vector computation
    test_simple_computation(State1),
    
    % Test 2: Matrix computation
    test_matrix_computation(State1),
    
    % Test 3: Error handling
    test_error_handling(State1),
    
    % Cleanup
    {ok, _} = dev_gpu_nif:terminate(State1, #{}, #{}),
    io:format("Tests completed~n").

test_simple_computation(State) ->
    io:format("Testing simple vector computation...~n"),
    InputData = <<1.0:32/float, 2.0:32/float, 3.0:32/float>>,
    Params = <<0.5:32/float, 0.1:32/float>>,
    {ok, Result} = dev_gpu_nif:compute(State, #{<<"input">> => InputData, <<"params">> => Params}, #{}),
    io:format("Result: ~p~n", [Result]).

test_matrix_computation(State) ->
    io:format("Testing matrix computation...~n"),
    Matrix = lists:flatten([[I*4 + J || J <- lists:seq(1,4)] || I <- lists:seq(0,3)]),
    InputData = list_to_binary([<<X:32/float>> || X <- Matrix]),
    Params = <<0.1:32/float, 0.2:32/float, 0.3:32/float>>,
    {ok, Result} = dev_gpu_nif:compute(State, #{<<"input">> => InputData, <<"params">> => Params}, #{}),
    io:format("Result: ~p~n", [Result]).

test_error_handling(State) ->
    io:format("Testing error handling...~n"),
    try
        dev_gpu_nif:compute(State, #{<<"input">> => <<"invalid">>}, #{})
    catch
        error:Reason ->
            io:format("Caught expected error: ~p~n", [Reason])
    end. 