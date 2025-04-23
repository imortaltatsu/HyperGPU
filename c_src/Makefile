ERLANG_PATH = $(shell erl -eval 'io:format("~s", [lists:concat([code:root_dir(), "/erts-", erlang:system_info(version), "/include"])])' -s init stop -noshell)
CUDA_PATH = /usr/local/cuda
CUDNN_PATH = /usr/local/cuda

CFLAGS = -fPIC -I$(ERLANG_PATH) -I$(CUDA_PATH)/include -I$(CUDNN_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/lib64 -L$(CUDNN_PATH)/lib64 -lcudart -lcudnn

all: gpu_nif.so

gpu_nif.so: gpu_nif.cpp
	g++ $(CFLAGS) -shared -o $@ $< $(LDFLAGS)

clean:
	rm -f gpu_nif.so 