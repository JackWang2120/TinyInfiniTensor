#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/unary.h"

#include "test.h"

namespace infini
{
    TEST(Allocator, testAlloc)
    {
        Shape shape = Shape{1, 2, 2, 3};
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Tensor a = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor b = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor c = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor d = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Allocator allocator = Allocator(runtime);
        //std::cout << "hello fuck" << std::endl;
        //std::cout << "Tensor a: " << a.toString() << std::endl;
        // allocate a->b->c
        size_t offsetA = allocator.alloc(a->getBytes());
        size_t offsetB = allocator.alloc(b->getBytes());
        size_t offsetC = allocator.alloc(c->getBytes());
        std::cout << "Tensor a:" << a->getBytes() << std::endl;
        std::cout << "Tensor b:" << b->getBytes() << std::endl;
        std::cout << "Tensor c:" << c->getBytes() << std::endl;
        std::cout << "Tensor d:" << d->getBytes() << std::endl;
        // free b, then allocate d
        allocator.free(offsetB, b->getBytes());
        size_t offsetD = allocator.alloc(d->getBytes());
        // expected to be a->d->c
        EXPECT_EQ(offsetB, offsetD);
        std::cout << "offsetA: " << offsetA << std::endl;
        std::cout << "offsetB: " << offsetB << std::endl;
        std::cout << "offsetC: " << offsetC << std::endl;
        std::cout << "offsetD: " << offsetD << std::endl;
        ASSERT_FALSE(offsetA == 0 && offsetB == 0 && offsetC == 0 && offsetD == 0);
    }
    

    TEST(Allocator, testAllocWithEndFreeBlock)
    {
        Shape shape = Shape{1, 2, 2, 3};
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Tensor a = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor b = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor c = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor d =
            make_ref<TensorObj>(Shape{2, 2, 2, 3}, DataType::Float32, runtime);
        Allocator allocator = Allocator(runtime);
        // allocate a->b->c
        allocator.alloc(a->getBytes());
        allocator.alloc(b->getBytes());
        size_t offsetC = allocator.alloc(c->getBytes());
        allocator.info();
        // free c, then allocate d
        allocator.free(offsetC, c->getBytes());
        std::cout << "Tensor d:" << d->getBytes() << std::endl;
        size_t offsetD = allocator.alloc(d->getBytes());
        allocator.info();
        // expected to be a->b->d, with no free block between b and c
        EXPECT_EQ(offsetC, offsetD);
    }

    TEST(Allocator, testGetPtr)
    {
        Shape shape = Shape{1, 2, 2, 3};
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Tensor a = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor b = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor c = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor d = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Allocator allocator = Allocator(runtime);
        // allocate a->b->c->d
        allocator.alloc(a->getBytes());
        allocator.alloc(b->getBytes());
        allocator.alloc(c->getBytes());
        allocator.alloc(d->getBytes());
        // multiple calls to the getPtr() function should return the same pointer
        void *ptr1 = allocator.getPtr();
        void *ptr2 = allocator.getPtr();
        EXPECT_EQ(ptr1, ptr2);
    }

} // namespace infini
