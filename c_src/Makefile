ERLANG_PATH = $(shell erl -eval 'io:format("~s", [lists:concat([code:root_dir(), "/erts-", erlang:system_info(version), "/include"])])' -s init stop -noshell)
CUDA_PATH = /usr/local/cuda
CUDNN_PATH = /usr/local/cudnn

ifeq ($(OS),Windows_NT)
    CFLAGS = -I"$(ERLANG_PATH)" -I"$(CUDA_PATH)/include" -I"$(CUDNN_PATH)/include" -fPIC
    LDFLAGS = -L"$(CUDA_PATH)/lib/x64" -L"$(CUDNN_PATH)/lib" -lcudart -lcudnn
    EXT = dll
else
    CFLAGS = -fPIC -I$(ERLANG_PATH) -I$(CUDA_PATH)/include -I$(CUDNN_PATH)/include
    LDFLAGS = -L$(CUDA_PATH)/lib64 -L$(CUDNN_PATH)/lib64 -lcudart -lcudnn
    EXT = so
endif

all: gpu_nif.$(EXT)

gpu_nif.$(EXT): gpu_nif.cpp
	g++ $(CFLAGS) -shared -o $@ $< $(LDFLAGS)

clean:
	rm -f gpu_nif.$(EXT) 