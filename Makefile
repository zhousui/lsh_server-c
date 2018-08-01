INC_DIR=./falconn

ALL_HEADERS = $(INC_DIR)/core/lsh_table.h $(INC_DIR)/core/cosine_distance.h $(INC_DIR)/core/euclidean_distance.h $(INC_DIR)/core/composite_hash_table.h $(INC_DIR)/core/stl_hash_table.h $(INC_DIR)/core/polytope_hash.h $(INC_DIR)/core/flat_hash_table.h $(INC_DIR)/core/probing_hash_table.h $(INC_DIR)/core/hyperplane_hash.h $(INC_DIR)/core/heap.h $(INC_DIR)/core/prefetchers.h $(INC_DIR)/core/incremental_sorter.h $(INC_DIR)/core/lsh_function_helpers.h $(INC_DIR)/core/hash_table_helpers.h $(INC_DIR)/core/data_storage.h $(INC_DIR)/core/nn_query.h $(INC_DIR)/lsh_nn_table.h $(INC_DIR)/wrapper/cpp_wrapper_impl.h $(INC_DIR)/falconn_global.h  $(INC_DIR)/core/data_transformation.h $(INC_DIR)/core/bit_packed_vector.h $(INC_DIR)/core/bit_packed_flat_hash_table.h $(INC_DIR)/core/random_projection_sketches.h $(INC_DIR)/experimental/pipes.h $(INC_DIR)/experimental/code_generation.h

CPP=gcc
CXX=g++
CXXFLAGS=-std=c++11 -Wall -O3 -g -march=native -I. -Ilog4cxx/include -I msgpack-c/include -I falconn -I./include -I./cnpy -I external/eigen -I falconn/include -I external/simple-serializer -I external/nlohmann
C_FLAGS= -std=c99

all:lsh_server

clean:
	rm -rf obj
	rm -rf lsh_server

obj/lsh_server.o: ./lsh_server.cpp
	mkdir -p obj
	$(CXX)  $(CXXFLAGS) -c -o obj/lsh_server.o ./lsh_server.cpp

obj/msgpack.o: ./msgpack.cpp
	mkdir -p obj
	$(CXX)  $(CXXFLAGS) -c -o obj/msgpack.o ./msgpack.cpp

lsh_server: obj/lsh_server.o obj/msgpack.o
	$(CXX) $(CXXFLAGS) -o lsh_server obj/lsh_server.o obj/msgpack.o -static -L./lib -lstdc++ -lzmq -lcnpy -llog4cxx -lapr-1 -laprutil-1 -lexpat -lz -pthread
