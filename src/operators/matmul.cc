#include "operators/matmul.h"
#include "utils/operator_utils.h"
namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        auto A = inputs[0];
        auto B = inputs[1];
        auto shapeA = A->getDims();
        auto shapeB = B->getDims();
        if (shapeA.size() < 2 || shapeB.size() < 2)
        {
            return std::nullopt;
        }
        if (transA)
        {
            std::swap(shapeA[shapeA.size() - 1], shapeA[shapeA.size() - 2]);
        }
        if (transB)
        {
            std::swap(shapeB[shapeB.size() - 1], shapeB[shapeB.size() - 2]);
        }
        if (shapeA[shapeA.size() - 1] != shapeB[shapeB.size() - 2])
        {
            return std::nullopt;
        }
        shapeA[shapeA.size() - 1] = 1;
        shapeB[shapeB.size() - 2] = 1;
        auto ans = infer_broadcast(shapeA, shapeB);
        return vector<Shape>{ans};
       // return std::nullopt;
    }

} // namespace infini