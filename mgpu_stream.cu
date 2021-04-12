#include <cstdint>
#include <iostream>
#include "helpers.cuh"
#include "encryption.cuh"

void encrypt_cpu(uint64_t * data, uint64_t num_entries, 
                 uint64_t num_iters, bool parallel=true) {

    #pragma omp parallel for if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++)
        data[entry] = permute64(entry, num_iters);
}

__global__ 
void decrypt_gpu(uint64_t * data, uint64_t num_entries, 
                 uint64_t num_iters) {

    const uint64_t thrdID = blockIdx.x*blockDim.x+threadIdx.x;
    const uint64_t stride = blockDim.x*gridDim.x;

    for (uint64_t entry = thrdID; entry < num_entries; entry += stride)
        data[entry] = unpermute64(data[entry], num_iters);
}

bool check_result_cpu(uint64_t * data, uint64_t num_entries,
                      bool parallel=true) {

    uint64_t counter = 0;

    #pragma omp parallel for reduction(+: counter) if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++)
        counter += data[entry] == entry;

    return counter == num_entries;
}

int main (int argc, char * argv[]) {

    const char * encrypted_file = "/dli/task/encrypted";

    Timer timer;

    const uint64_t num_entries = 1UL << 26;
    const uint64_t num_iters = 1UL << 10;
    const bool openmp = true;

int num_gpus;
cudaGetDeviceCount(&num_gpus);

const uint64_t num_streams = 32;
cudaStream_t streams[num_gpus][num_streams];

const uint64_t stream_chunk_size = sdiv(sdiv(num_entries, num_gpus),num_streams);
const uint64_t gpu_chunk_size = num_streams * stream_chunk_size;


timer.start();
for(int gpu=0; gpu<num_gpus; gpu++)
{
    cudaSetDevice(gpu);
    for(int stream=0; stream<num_streams; stream++ )
    {
        cudaStreamCreate(&streams[gpu][stream]);
    }
}
timer.stop("create streams");

    uint64_t * data_cpu, * data_gpu[num_gpus];
    cudaMallocHost(&data_cpu, sizeof(uint64_t)*num_entries);
    
timer.start();
for(int gpu=0; gpu<num_gpus; gpu++)
{
    cudaSetDevice(gpu);
 
    const uint64_t lower = gpu_chunk_size * gpu;
    const uint64_t upper = min(lower+gpu_chunk_size, num_entries);
    const uint64_t width = upper - lower;
    
    cudaMalloc(&data_gpu[gpu], sizeof(uint64_t)*width);
    check_last_error();
    
}
timer.stop("malloc gpu memory");

    if (!encrypted_file_exists(encrypted_file)) {
        encrypt_cpu(data_cpu, num_entries, num_iters, openmp);
        write_encrypted_to_file(encrypted_file, data_cpu, sizeof(uint64_t)*num_entries);
    } else {
        read_encrypted_from_file(encrypted_file, data_cpu, sizeof(uint64_t)*num_entries);
    }

    timer.start();
for(int gpu=0; gpu<num_gpus; gpu++)
{
    cudaSetDevice(gpu);
    for(int stream=0; stream<num_streams; stream++ )
    {  
        const uint64_t stream_offset = stream_chunk_size * stream;
         
        const uint64_t lower = gpu_chunk_size * gpu + stream_offset;
        const uint64_t upper = min(lower+stream_chunk_size, num_entries);
        const uint64_t width = upper - lower;    
        
        cudaMemcpyAsync(data_gpu[gpu]+stream_offset, data_cpu+lower, 
               sizeof(uint64_t)*width, cudaMemcpyHostToDevice, streams[gpu][stream]);
        check_last_error();
        
        
        decrypt_gpu<<<80*32, 64,0,streams[gpu][stream] >>>(data_gpu[gpu]+stream_offset, width, num_iters);
        check_last_error();
        
        cudaMemcpyAsync(data_cpu+lower, data_gpu[gpu]+stream_offset, 
               sizeof(uint64_t)*width, cudaMemcpyDeviceToHost, streams[gpu][stream]);
        
    }
}

for (uint64_t gpu = 0; gpu < num_gpus; gpu++) {
    cudaSetDevice(gpu);
    for (uint64_t stream = 0; stream < num_streams; stream++) {
        cudaStreamSynchronize(streams[gpu][stream]);
    }
}
    timer.stop("total time on GPU");
    check_last_error();

    const bool success = check_result_cpu(data_cpu, num_entries, openmp);
    std::cout << "STATUS: test " 
              << ( success ? "passed" : "failed")
              << std::endl;

    cudaFreeHost(data_cpu);
    
    
 timer.start();
for(int gpu=0; gpu<num_gpus; gpu++)
{
    cudaSetDevice(gpu);
    for(int stream=0; stream<num_streams; stream++ )
    {
        cudaStreamDestroy(streams[gpu][stream]);
    }
}

for(int gpu=0; gpu<num_gpus; gpu++)
{
    cudaFree(data_gpu[gpu]);
}

check_last_error();
timer.stop("free");   
    
    
}
