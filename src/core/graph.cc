#include "core/graph.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <queue>
#include <string>

namespace infini {

void GraphObj::addOperatorAndConnect(const Operator &op) {
  sorted = false;
  ops.push_back(op);
  for (auto &input : op->getInputs()) {
    if (input) {
      input->addTarget(op);
      if (auto pred = input->getSource()) {
        pred->addSuccessors(op);
        op->addPredecessors(pred);
      }
    }
  }
  for (auto &output : op->getOutputs()) {
    if (output) {
      output->setSource(op);
      for (auto &succ : output->getTargets()) {
        succ->addPredecessors(op);
        op->addSuccessors(succ);
      }
    }
  }
}

string GraphObj::toString() const {
  std::cout << "===========GraphObj::toString()===========" << std::endl;
  std::ostringstream oss;
  oss << "Graph Tensors:\n";
  std::cout << "===========Before 遍历 Tensors===========" << std::endl;
  for (const auto &tensor : tensors)
    oss << tensor << "\n";

  oss << "Graph operators:\n";
  std::cout << "===========Before 遍历 ops===========" << std::endl;
  for (const auto &op : ops) {
    std::cout << "正在遍历算子" << (op->toString()) << std::endl;
    vector<UidBaseType> preds, succs;
    for (auto &o : op->getPredecessors())
      preds.emplace_back(o->getGuid());
    for (auto &o : op->getSuccessors())
      succs.emplace_back(o->getGuid());
    oss << "OP " << op->getGuid();
    oss << ", pred " << vecToString(preds);
    oss << ", succ " << vecToString(succs);
    oss << ", " << op << "\n";
  }
  std::cout << "===========After 遍历 ops===========" << std::endl;
  return oss.str();
}

bool GraphObj::topo_sort() {
  if (this->sorted) {
    return true;
  }
  std::vector<Operator> sorted;
  std::unordered_set<OperatorObj *> flags;
  sorted.reserve(ops.size());
  flags.reserve(ops.size());
  while (sorted.size() < ops.size()) {
    // Any node is move to sorted in this loop.
    auto modified = false;
    for (auto const &op : ops) {
      if (auto const &inputs = op->getInputs();
          flags.find(op.get()) == flags.end() &&
          std::all_of(inputs.begin(), inputs.end(),
                      [&flags](auto const &input) {
                        auto ptr = input->getSource().get();
                        return !ptr || flags.find(ptr) != flags.end();
                      })) {
        modified = true;
        sorted.emplace_back(op);
        flags.insert(op.get());
      }
    }
    if (!modified) {
      return false;
    }
  }
  this->ops = std::move(sorted);
  return this->sorted = true;
}
bool isSamePermute(const TransposeObj *op1, const TransposeObj *op2) {
  auto perm1 = op1->getPermute();
  auto perm2 = op2->getPermute();
  if (perm1.size() != perm2.size()) {
    return false;
  }
  for (int i = 0; i < (int)perm1.size(); i++) {
    if (perm1[i] != perm2[i]) {
      return false;
    }
  }
  return true;
}
bool isTransposeLast2dim(const TransposeObj *op) {
  std::cout << "===========isTransposeLast2dim===========" << std::endl;
  auto perm = op->getPermute();
  std::swap(perm[perm.size() - 1], perm[perm.size() - 2]);
  for (int i = 0; i < (int)perm.size(); i++) {
    if (perm[i] != i) {
      return false;
    }
  }
  return true;
}
void GraphObj::operatorMerge(Tensor& input, vector<Operator> &removed_ops, vector<Tensor> &removed_tensors, Operator& op) {
  std::cout << "===========operatorMerge===========" << std::endl;
  if(input->getSource()){
        if (input->getSource()->getOpType() == OpType::Transpose &&
            isTransposeLast2dim(
                static_cast<const TransposeObj *>(input->getSource().get()))) {
          std::cout << "===========optimize:input2可以进行算子融合==========="
                    << std::endl;
          auto transposeOp = input->getSource();
          // 判断是否可以合并(前驱算子的输入是不是transpose,且对最后两个维度做交换)

          // 获取前驱算子的输入tensor：transposeInput，输出tensor就是input1
          auto transposeInput = transposeOp->getInputs(0);

          op->replaceInput(input, transposeInput);
          op->removePredecessors(transposeOp);
          transposeInput->removeTarget(transposeOp);
          transposeInput->addTarget(op);
         // removeTensor(input2);
          removed_tensors.emplace_back(input);
          static_cast<MatmulObj *>(op.get())->setTransB(
              !static_cast<MatmulObj *>(op.get())->getTransB());
          // 把transposeOp从图中删除
          //ops.erase(std::find(ops.begin(), ops.end(), transposeOp));
          removed_ops.emplace_back(transposeOp);
        }
      }
}
void GraphObj::optimize() {
  // =================================== 作业
  // ===================================
  // TODO: 设计一个算法来实现指定的图优化规则
  // 图优化规则如下：
  // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose
  // 算子，且做的是相反的操作，可以将其全部删除）
  // 2.
  // 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
  // =================================== 作业
  // ===================================
  std::cout << "optimize start" << std::endl;
  vector<Operator> removed_ops;
  vector<Tensor> removed_tensors;
  IT_ASSERT(topo_sort() == true);
  for (auto it = ops.begin(); it != ops.end();) {
    auto op = *it;
    std::cout << "===========optimize:for1===========" << std::endl;
    if (op->getOpType() == OpType::Transpose) {
      std::cout << "===========optimize:op->getOpType() == "
                   "OpType::Transpose==========="
                << std::endl;
      auto input = op->getInputs(0);
      auto output = op->getOutput();
      if (input->getTargets().size() == 1 && output->getTargets().size() == 1) {
        std::cout << "===========optimize:input->getTargets().size() == 1 && "
                     "output->getTargets().size() == 1==========="
                  << std::endl;
        auto nextOp = output->getTargets()[0];
        if (nextOp->getOpType() == OpType::Transpose) {
          std::cout << "===========optimize:nextOp->getOpType() == "
                       "OpType::Transpose==========="
                    << std::endl;
          auto nextInput = nextOp->getInputs(0);
          // 可以去除冗余算子
          if (nextInput == output &&
              isSamePermute(static_cast<const TransposeObj *>(op.get()),
                            static_cast<const TransposeObj *>(nextOp.get()))) {
            std::cout << "===========optimize:去除冗余算子==========="
                      << std::endl;
            // remove op and nextOp from graph
            auto opPre = op->getPredecessors();
            auto nextOpSucc = nextOp->getSuccessors();
            // 只考虑单输入的情况
            auto tensor = op->getInputs()[0];
            auto nextTensor = nextOp->getOutputs()[0];
            tensor->removeTarget(op);

            // if (opPre.size() == 1) {
            //   std::cout <<
            //   "===========optimize:只考虑单输入的情况==========="
            //             << std::endl;
            //   // 改变算子的前驱和后继
            //   // 改变tensor的前驱和后继

            //   opPre[0]->removeSuccessors(op);
            //   for (auto &succ : nextOpSucc) {
            //     succ->removePredecessors(nextOp);
            //     opPre[0]->addSuccessors(succ);
            //     succ->addPredecessors(opPre[0]);
            //     tensor->addTarget(succ);
            //     succ->replaceInput(succ->getInputs()[0], tensor);
            //   }

            //   // nextOpSucc[0]->replaceInput(nextOpSucc[0]->getInputs()[0],
            //   // tensor);
            // }else{
            std::cout << "===========optimize:只考虑单输入的情况==========="
                      << std::endl;
            // 改变算子的前驱和后继
            // 改变tensor的前驱和后继
            for (auto &succ : nextOpSucc) {
              succ->removePredecessors(nextOp);
              for (auto &pre : opPre) {
                pre->addSuccessors(succ);
                succ->addPredecessors(pre);
              }
              tensor->addTarget(succ);
              succ->replaceInput(succ->getInputs()[0], tensor);
            }
            // }
            // 删除tensor
            std::cout << "===========optimize:删除tensor==========="
                      << std::endl;
            std::cout << "正在删除tensor" << output->toString() << std::endl;
            //removeTensor(output);
            removed_tensors.emplace_back(output);
            std::cout << "正在删除tensor" << nextTensor->toString()
                      << std::endl;
            //removeTensor(nextTensor);
            removed_tensors.emplace_back(nextTensor);
            // 删除算子
            std::cout << "===========optimize:删除算子===========" << std::endl;
            std::cout << "正在删除算子" << (*it)->toString() << std::endl;
            removed_ops.emplace_back(*it);
            it++;
            //it = ops.erase(it);
            std::cout << "正在删除算子" << (*it)->toString() << std::endl;
            removed_ops.emplace_back(*it);
            it++;
            //it = ops.erase(it);
            continue;
          }
        }
      }
    }
    it++;
  }
  // 合并算子
  for (auto it = ops.begin(); it != ops.end(); it++) {
    std::cout << "===========optimize:for2===========" << std::endl;
    auto op = *it;
    if (op->getOpType() == OpType::MatMul) {
      std::cout << "===========optimize:op->getOpType() == "
                   "OpType::MatMul==========="
                << std::endl;
      auto input1 = op->getInputs(0);
      auto input2 = op->getInputs(1);
      auto output = op->getOutput();
      // 判断前驱算子是不是transpose
      std::cout << "判断前驱算子能否融合" << std::endl;
      operatorMerge(input1, removed_ops, removed_tensors,op);
      operatorMerge(input2, removed_ops, removed_tensors,op);
      // if (input1->getSource()) {
      //   if (input1->getSource()->getOpType() == OpType::Transpose &&
      //       isTransposeLast2dim(
      //           static_cast<const TransposeObj *>(input1->getSource().get()))) {
      //     std::cout << "===========optimize:input1可以进行算子融合==========="
      //               << std::endl;
      //     auto transposeOp = input1->getSource();
      //     // 判断是否可以合并(前驱算子的输入是不是transpose,且对最后两个维度做交换)

      //     // 获取前驱算子的输入tensor：transposeInput，输出tensor就是input1
      //     auto transposeInput = transposeOp->getInputs(0);

      //     op->replaceInput(input1, transposeInput);
      //     removeTensor(input1);
      //     static_cast<MatmulObj *>(op.get())->setTransA(
      //         !static_cast<MatmulObj *>(op.get())->getTransA());
      //     // 把transposeOp从图中删除
      //     ops.erase(std::find(ops.begin(), ops.end(), transposeOp));
      //   }
      // }
      // if(input2->getSource()){
      //   if (input2->getSource()->getOpType() == OpType::Transpose &&
      //       isTransposeLast2dim(
      //           static_cast<const TransposeObj *>(input2->getSource().get()))) {
      //     std::cout << "===========optimize:input2可以进行算子融合==========="
      //               << std::endl;
      //     auto transposeOp = input2->getSource();
      //     // 判断是否可以合并(前驱算子的输入是不是transpose,且对最后两个维度做交换)

      //     // 获取前驱算子的输入tensor：transposeInput，输出tensor就是input1
      //     auto transposeInput = transposeOp->getInputs(0);

      //     op->replaceInput(input2, transposeInput);
      //    // removeTensor(input2);
      //     removed_tensors.emplace_back(input2);
      //     static_cast<MatmulObj *>(op.get())->setTransB(
      //         !static_cast<MatmulObj *>(op.get())->getTransB());
      //     // 把transposeOp从图中删除
      //     //ops.erase(std::find(ops.begin(), ops.end(), transposeOp));
      //     removed_ops.emplace_back(transposeOp);
      //   }
      // }
    }
  }
  for (auto &op : removed_ops) {
    removeOperator(op);
  }
  for (auto &tensor : removed_tensors) {
    removeTensor(tensor);
  }
  std::cout << "ops.size = " << ops.size() << std::endl;
  std::cout << "tensors.size = " << tensors.size() << std::endl;
  std::cout << "optimize end" << std::endl;
  std::cout << toString() << std::endl;
}

Tensor GraphObj::getTensor(int fuid) const {
  for (auto tensor : tensors) {
    if (tensor->getFuid() == fuid) {
      return tensor;
    }
  }
  return nullptr;
}

void GraphObj::shape_infer() {
  for (auto &op : ops) {
    auto ans = op->inferShape();
    IT_ASSERT(ans.has_value());
    auto oldOutputs = op->getOutputs();
    IT_ASSERT(ans.value().size() == oldOutputs.size());
    // replace the old outputshape and size with new one
    for (int i = 0; i < (int)ans.value().size(); ++i) {
      auto newShape = ans.value()[i];
      auto oldShape = oldOutputs[i]->getDims();
      auto fuid = oldOutputs[i]->getFuid();
      if (newShape != oldShape) {
        auto tensor = this->getTensor(fuid);
        tensor->setShape(newShape);
      }
    }
  }
}

void GraphObj::dataMalloc() {
  // topological sorting first
  IT_ASSERT(topo_sort() == true);

  // =================================== 作业
  // ===================================
  // TODO：利用 allocator 给计算图分配内存
  // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor
  // 绑定内存
  // =================================== 作业
  // =================================== for(auto& tensor : tensors) {
  //     size_t offset = allocator.alloc(tensor->getBytes());
  //     tensor->setDataBlob(make_ref<BlobObj>(runtime,
  //     static_cast<void*>(static_cast<char*>(allocator.getPtr()) + offset)));
  // }
  size_t total_size = 0;
  for (auto &tensor : tensors) {
    total_size += tensor->getBytes();
  }
  size_t addr = allocator.alloc(total_size);
  for (auto &tensor : tensors) {
    auto ptr = static_cast<char *>(allocator.getPtr()) + addr;
    tensor->setDataBlob(make_ref<BlobObj>(runtime, ptr));
    addr += tensor->getBytes();
  }
  allocator.info();
}

Tensor GraphObj::addTensor(Shape dim, DataType dtype) {
  return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
}

Tensor GraphObj::addTensor(const Tensor &tensor) {
  IT_ASSERT(tensor->getRuntime() == runtime,
            std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                tensor->getRuntime()->toString() + " to " +
                runtime->toString());
  tensors.emplace_back(tensor);
  return tensor;
}

TensorVec GraphObj::addTensor(const TensorVec &tensors) {
  for (auto &t : tensors)
    addTensor(t);
  return tensors;
}

// tensor's "source" and "target" must be in "ops".
// tensor has no "source" and no "target" must not exist.
// "inputs" or "outputs" of operators must be in "tensors"
// "predecessors" and "successors" of an operator of "ops" must be in "ops".
bool GraphObj::checkValid() const {
  for (auto tensor : tensors) {
    IT_ASSERT(
        !(tensor->getTargets().size() == 0 && nullptr == tensor->getSource()));
    for (auto op : tensor->getTargets()) {
      IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
    }
    auto op = tensor->getSource();
    IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
  }
  for (auto op : ops) {
    for (auto tensor : op->getInputs()) {
      IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                tensors.end());
    }
    for (auto tensor : op->getOutputs()) {
      IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                tensors.end());
    }
    for (auto pre : op->getPredecessors()) {
      IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
    }
    for (auto suc : op->getSuccessors()) {
      IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
    }
  }
  std::set<UidBaseType> s;
  // check whether two tensors with the same FUID exist
  for (auto tensor : tensors) {
    int cnt = s.count(tensor->getFuid());
    IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
    s.insert(tensor->getFuid());
  }
  return true;
}

} // namespace infini