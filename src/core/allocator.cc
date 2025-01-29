#include "core/allocator.h"
#include <cmath>
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;
        maxSize = 1ULL<<32;
        remainSize = maxSize;
        //free_blocks.insert({0, maxSize});
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
                    std::cout << "in it->second == size" << std::endl;
                }else{
                    size_t newSize = it->second - size;
                    size_t new_addr = addr + size;
                    free_blocks.erase(it);
                    free_blocks.insert({new_addr, newSize});
                }
                this->remainSize -= size;
                return addr;
            }
        }
        //  std::cout << "free_blocks:" << std::endl;
        // for(auto it = free_blocks.begin(); it != free_blocks.end(); it++) {
        //    std::cout<<it->first<<" "<<it->second<<std::endl;
        // }
        //空闲块里面没有足够的空间，直接在末尾分配空闲空间
        size_t addr = maxSize - remainSize;
        remainSize -= size;
        if(remainSize < 0) {
            throw std::runtime_error("No extra free space available for allocation");
        }
       return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        this->used -= size;
        free_blocks.insert({addr, size});
        auto it = free_blocks.find(addr);
        
        //和后面的空闲块合并
        auto itr = std::next(it);
        if(itr != free_blocks.end() && itr->first == it->first + it->second) {
            it->second += itr->second;
            free_blocks.erase(itr);
        }
        //和前面的空闲块合并
        if(it != free_blocks.begin()) {
            auto itp = std::prev(it);
            if(itp->first + itp->second == it->first) {
                addr = itp->first;
                size += itp->second;
                free_blocks.erase(itp);
                free_blocks.erase(it);
                free_blocks.insert({addr, size});
            }
        }
        //如果是最后一个空闲块，直接合并
        if(it->first + it->second == maxSize - remainSize) {
            remainSize += it->second;
            free_blocks.erase(it);
        }
       

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
