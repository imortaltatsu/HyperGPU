%%% @doc A device that provides deterministic GPU computations using CUDA/CuDNN.
%%% This device uses NIFs to interface with the GPU hardware.
-module(dev_gpu_nif).
-export([info/2, init/3, compute/3, terminate/3, snapshot/3, normalize/3]).
-export([load/0, unload/0]).

%% NIF functions
-export([
    gpu_init/0,
    gpu_compute/2,
    gpu_terminate/0,
    gpu_get_state/0,
    gpu_set_state/1
]).

-on_load(load/0).

-include("include/hb.hrl").

%% NIF stubs
gpu_init() -> erlang:nif_error(nif_not_loaded).
gpu_compute(_, _) -> erlang:nif_error(nif_not_loaded).
gpu_terminate() -> erlang:nif_error(nif_not_loaded).
gpu_get_state() -> erlang:nif_error(nif_not_loaded).
gpu_set_state(_) -> erlang:nif_error(nif_not_loaded).

%% @doc Load the NIF library
load() ->
    PrivDir = case code:priv_dir(?MODULE) of
        {error, _} ->
            EbinDir = filename:dirname(code:which(?MODULE)),
            AppPath = filename:dirname(EbinDir),
            filename:join(AppPath, "priv");
        Path -> Path
    end,
    erlang:load_nif(filename:join(PrivDir, "gpu_nif"), 0).

%% @doc Unload the NIF library
unload() ->
    gpu_terminate(),
    ok.

%% @doc Device info
info(_Msg1, _Opts) ->
    #{
        excludes => []
    }.

%% @doc Initialize the GPU device
init(M1, _M2, Opts) ->
    ?event(running_init),
    case gpu_init() of
        {ok, _} -> 
            ?event(gpu_initialized),
            {ok, M1};
        {error, Reason} ->
            throw({gpu_init_error, Reason})
    end.

%% @doc Perform GPU computation
compute(M1, M2, Opts) ->
    ?event(running_compute),
    try
        % Get input data from message
        InputData = hb_ao:get(<<"input">>, M2, Opts),
        % Get computation parameters
        Params = hb_ao:get(<<"params">>, M2, Opts),
        
        % Perform GPU computation
        case gpu_compute(InputData, Params) of
            {ok, Result} ->
                ?event(computation_completed),
                {ok, M1#{<<"result">> => Result}};
            {error, Reason} ->
                throw({gpu_compute_error, Reason})
        end
    catch
        error:Reason ->
            throw({gpu_compute_error, Reason})
    end.

%% @doc Terminate the GPU device
terminate(M1, _M2, _Opts) ->
    ?event(terminating),
    case gpu_terminate() of
        ok -> {ok, M1};
        {error, Reason} -> throw({gpu_terminate_error, Reason})
    end.

%% @doc Get GPU state snapshot
snapshot(M1, _M2, _Opts) ->
    case gpu_get_state() of
        {ok, State} -> {ok, M1#{<<"state">> => State}};
        {error, Reason} -> throw({gpu_snapshot_error, Reason})
    end.

%% @doc Normalize GPU state
normalize(M1, M2, Opts) ->
    case hb_ao:get(<<"state">>, M2, Opts) of
        not_found -> {ok, M1};
        State ->
            case gpu_set_state(State) of
                ok -> {ok, M1};
                {error, Reason} -> throw({gpu_normalize_error, Reason})
            end
    end. 