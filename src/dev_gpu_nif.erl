%%% @doc A device that provides GPU-accelerated image generation using Stable Diffusion.
%%% This device uses NIFs to interface with the GPU hardware and stable-diffusion.cpp.
-module(dev_gpu_nif).
-export([info/0, info/2, init/3, compute/3, terminate/3, snapshot/3, normalize/3]).
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

%% @doc Basic device info
info() ->
    #{
        excludes => [],
        endpoints => #{
            <<"generate">> => fun generate/4,
            <<"init">> => fun init/4,
            <<"terminate">> => fun terminate/4,
            <<"snapshot">> => fun snapshot/4,
            <<"normalize">> => fun normalize/4
        }
    }.

%% @doc Device info with message context
info(_Msg1, _Opts) ->
    info().

%% @doc Generate image endpoint
generate(M1, M2, Opts, _) ->
    compute(M1, M2, Opts).

%% @doc Initialize the GPU device and Stable Diffusion
init(M1, _M2, _Opts, _) ->
    ?event(running_init),
    try
        case gpu_init() of
            {ok, _} -> 
                ?event(gpu_initialized),
                {ok, M1};
            {error, Reason} ->
                ?event(gpu_init_error, {reason, Reason}),
                throw({gpu_init_error, Reason})
        end
    catch
        Class:Reason:Stack ->
            ?event(gpu_init_error, {class, Class}, {reason, Reason}, {stack, Stack}),
            throw({gpu_init_error, Reason})
    end.

%% @doc Generate an image using Stable Diffusion
%% The input message should contain:
%% - prompt: The text prompt for image generation
%% - params: Optional parameters for generation
%%   - width: Image width (default: 512)
%%   - height: Image height (default: 512)
%%   - steps: Number of sampling steps (default: 20)
%%   - seed: Random seed (-1 for random)
%%   - cfg_scale: Classifier-free guidance scale (default: 7.0)
%%   - sampler: Sampling method (default: euler_a)
%%   - schedule: Denoiser schedule (default: discrete)
compute(M1, M2, Opts) ->
    ?event(running_compute),
    try
        % Get prompt from message
        Prompt = hb_ao:get(<<"prompt">>, M2, Opts),
        if not is_binary(Prompt) ->
            ?event(invalid_prompt, {prompt, Prompt}),
            throw({invalid_prompt, "Prompt must be a binary string"});
        true -> ok
        end,

        % Get optional parameters
        Params = maps:merge(#{
            <<"width">> => 512,
            <<"height">> => 512,
            <<"steps">> => 20,
            <<"seed">> => -1,
            <<"cfg_scale">> => 7.0,
            <<"sampler">> => <<"euler_a">>,
            <<"schedule">> => <<"discrete">>,
            <<"clip_skip">> => -1,
            <<"vae_tiling">> => false,
            <<"vae_on_cpu">> => false,
            <<"clip_on_cpu">> => false,
            <<"diffusion_fa">> => false
        }, hb_ao:get(<<"params">>, M2, #{}, Opts)),

        % Convert parameters to binary format
        ParamsBin = term_to_binary(Params),

        % Perform GPU computation
        case gpu_compute(Prompt, ParamsBin) of
            {ok, Result} ->
                ?event(computation_completed),
                {ok, M1#{
                    <<"result">> => Result,
                    <<"width">> => maps:get(<<"width">>, Params),
                    <<"height">> => maps:get(<<"height">>, Params)
                }};
            {error, Reason} ->
                ?event(gpu_compute_error, {reason, Reason}),
                throw({gpu_compute_error, Reason})
        end
    catch
        Class:Reason:Stack ->
            ?event(gpu_compute_error, {class, Class}, {reason, Reason}, {stack, Stack}),
            throw({gpu_compute_error, Reason})
    end.

%% @doc Terminate the GPU device
terminate(M1, _M2, _Opts, _) ->
    ?event(terminating),
    try
        case gpu_terminate() of
            ok -> {ok, M1};
            {error, Reason} ->
                ?event(gpu_terminate_error, {reason, Reason}),
                throw({gpu_terminate_error, Reason})
        end
    catch
        Class:Reason:Stack ->
            ?event(gpu_terminate_error, {class, Class}, {reason, Reason}, {stack, Stack}),
            throw({gpu_terminate_error, Reason})
    end.

%% @doc Get GPU state snapshot
snapshot(M1, _M2, _Opts, _) ->
    ?event(getting_snapshot),
    try
        case gpu_get_state() of
            {ok, State} -> {ok, M1#{<<"state">> => State}};
            {error, Reason} ->
                ?event(gpu_snapshot_error, {reason, Reason}),
                throw({gpu_snapshot_error, Reason})
        end
    catch
        Class:Reason:Stack ->
            ?event(gpu_snapshot_error, {class, Class}, {reason, Reason}, {stack, Stack}),
            throw({gpu_snapshot_error, Reason})
    end.

%% @doc Normalize GPU state
normalize(M1, M2, Opts, _) ->
    ?event(normalizing_state),
    try
        case hb_ao:get(<<"state">>, M2, Opts) of
            not_found -> {ok, M1};
            State ->
                case gpu_set_state(State) of
                    ok -> {ok, M1};
                    {error, Reason} ->
                        ?event(gpu_normalize_error, {reason, Reason}),
                        throw({gpu_normalize_error, Reason})
                end
        end
    catch
        Class:Reason:Stack ->
            ?event(gpu_normalize_error, {class, Class}, {reason, Reason}, {stack, Stack}),
            throw({gpu_normalize_error, Reason})
    end. 