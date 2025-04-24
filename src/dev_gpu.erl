%%% @doc The hyperbeam GPU device, which provides GPU computation capabilities
%%% using CUDA and CuDNN. This device allows for high-performance computation
%%% on NVIDIA GPUs.
-module(dev_gpu).
-export([info/0, init/3, compute/3, terminate/3, get_state/3, set_state/3]).
-include("include/hb.hrl").

%% @doc Return the device's exported functions
info() ->
    #{
        default => fun compute/3,
        exports => [<<"init">>, <<"compute">>, <<"terminate">>, <<"get_state">>, <<"set_state">>]
    }.

%% @doc Initialize the GPU device
init(_, Request, _NodeMsg) ->
    case gpu_nif:gpu_init() of
        ok -> {ok, #{ <<"status">> => <<"initialized">> }};
        {error, Reason} -> {error, Reason}
    end.

%% @doc Perform GPU computation
compute(_, Request, _NodeMsg) ->
    Input = hb_ao:get(<<"input">>, Request),
    case gpu_nif:gpu_compute(Input) of
        {ok, Result} -> {ok, #{ <<"result">> => Result }};
        {error, Reason} -> {error, Reason}
    end.

%% @doc Terminate GPU resources
terminate(_, Request, _NodeMsg) ->
    case gpu_nif:gpu_terminate() of
        ok -> {ok, #{ <<"status">> => <<"terminated">> }};
        {error, Reason} -> {error, Reason}
    end.

%% @doc Get the current GPU state
get_state(_, Request, _NodeMsg) ->
    case gpu_nif:gpu_get_state() of
        {ok, State} -> {ok, #{ <<"state">> => State }};
        {error, Reason} -> {error, Reason}
    end.

%% @doc Set the GPU state
set_state(_, Request, _NodeMsg) ->
    State = hb_ao:get(<<"state">>, Request),
    case gpu_nif:gpu_set_state(State) of
        ok -> {ok, #{ <<"status">> => <<"state_set">> }};
        {error, Reason} -> {error, Reason}
    end. 