#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        this->used += size;
        this->peak = std::max(this->peak, this->used);
        for(auto it = free_blocks.begin(); it != free_blocks.end(); it++) {
            if(it->second >= size) {
                size_t addr = it->first;
                if(it->second == size) {
                    free_blocks.erase(it);
                    std::cout << "in it-<second == size" << std::endl;
                }else{
                    size_t newSize = it->second - size;
                    size_t new_addr = addr + size;
                    free_blocks.erase(it);
                    free_blocks.insert({new_addr, newSize});
                }
                return addr;
            }
        }
         std::cout << "free_blocks:" << std::endl;
        for(auto it = free_blocks.begin(); it != free_blocks.end(); it++) {
           std::cout<<it->first<<" "<<it->second<<std::endl;
        }
        size_t addr = reinterpret_cast<size_t>(runtime->alloc(size));
        
        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        if(free_blocks.find(addr + size) != free_blocks.end()) {
            size_t newSize = size + free_blocks[addr + size];
            free_blocks.erase(addr + size);
            free_blocks.insert({addr, newSize});
        }else{
            free_blocks.insert({addr, size});
        }
        std::cout << "free_blocks:" << std::endl;
        for(auto it = free_blocks.begin(); it != free_blocks.end(); it++) {
           std::cout<<it->first<<" "<<it->second<<std::endl;
        }
        std::cout << "size = " << size << std::endl;
        this->used -= size;
        std::cout << "this->used = " << this->used << std::endl; 
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
