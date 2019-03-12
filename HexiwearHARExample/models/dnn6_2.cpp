// Auto generated by utensor-cli

#include "dnn6_2_weight.hpp"
#include "dnn6_2.hpp"
#include "uTensor/ops/MatrixOps.hpp"
#include "uTensor/ops/MathOps.hpp"
#include "uTensor/core/context.hpp"
#include "uTensor/core/tensor.hpp"
#include "uTensor/ops/ArrayOps.hpp"
#include "uTensor/ops/NnOps.hpp"


void get_dnn6_2_ctx(Context& ctx, Tensor* input_0) {

{ // add tensor for placeholders
    ctx.add(input_0, "Placeholder:0", 2);
}
{    
    ctx.add(new BinaryTensor<int>({1}, inline_MatMul_eightbit_Placeholder__port__0_reshape_dims_0), 
            "MatMul_eightbit/Placeholder__port__0/reshape_dims:0", 
            1);
}
{
    ctx.add(new RamTensor<float>(), "MatMul_eightbit/Placeholder__port__0/reshape:0", 2);
    ctx.push(new ReshapeOp(), 
             { "Placeholder:0", "MatMul_eightbit/Placeholder__port__0/reshape_dims:0" },
             { "MatMul_eightbit/Placeholder__port__0/reshape:0" });
    ctx.eval();
}
{    
    ctx.add(new BinaryTensor<int>({1}, inline_MatMul_eightbit_Placeholder__port__0_reduction_dims_0), 
            "MatMul_eightbit/Placeholder__port__0/reduction_dims:0", 
            2);
}
{   
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "MatMul_eightbit/Placeholder__port__0/min:0", 1);
    ctx.push(new MinOp(), 
             { "MatMul_eightbit/Placeholder__port__0/reshape:0", "MatMul_eightbit/Placeholder__port__0/reduction_dims:0" },
             { "MatMul_eightbit/Placeholder__port__0/min:0" });
    ctx.eval();
}
{   
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "MatMul_eightbit/Placeholder__port__0/max:0", 1);
    ctx.push(new MaxOp(), 
             { "MatMul_eightbit/Placeholder__port__0/reshape:0", "MatMul_eightbit/Placeholder__port__0/reduction_dims:0" },
             { "MatMul_eightbit/Placeholder__port__0/max:0" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<uint8_t>(), "MatMul_eightbit/Placeholder__port__0/quantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "MatMul_eightbit/Placeholder__port__0/quantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "MatMul_eightbit/Placeholder__port__0/quantize:2", 1);
    ctx.push(new QuantizeV2Op(),
             {  "Placeholder:0",  "MatMul_eightbit/Placeholder__port__0/min:0", "MatMul_eightbit/Placeholder__port__0/max:0" },
             {  "MatMul_eightbit/Placeholder__port__0/quantize:0",  "MatMul_eightbit/Placeholder__port__0/quantize:1", "MatMul_eightbit/Placeholder__port__0/quantize:2" });
    ctx.eval();
}
{    
    ctx.add(new BinaryTensor<uint8_t>({768,300}, inline_Variable_quantized_const_0), 
            "Variable_quantized_const:0", 
            1);
}
{    
    ctx.add(new BinaryTensor<float>({}, inline_Variable_quantized_min_0), 
            "Variable_quantized_min:0", 
            1);
}
{    
    ctx.add(new BinaryTensor<float>({}, inline_Variable_quantized_max_0), 
            "Variable_quantized_max:0", 
            1);
}
{
    ctx.add(new RamTensor<int>(), "MatMul/eightbit:0", 2);
    ctx.add(new RamTensor<float>({1}), "MatMul/eightbit:1", 2);
    ctx.add(new RamTensor<float>({1}), "MatMul/eightbit:2", 2);
    ctx.push(new QntMatMulOp<uint8_t, uint8_t, int>(), 
             { "MatMul_eightbit/Placeholder__port__0/quantize:0", "MatMul_eightbit/Placeholder__port__0/quantize:1", "MatMul_eightbit/Placeholder__port__0/quantize:2", "Variable_quantized_const:0", "Variable_quantized_min:0",  "Variable_quantized_max:0" },
             { "MatMul/eightbit:0", "MatMul/eightbit:1",  "MatMul/eightbit:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<float>({1}), "MatMul/eightbit/requant_range:0", 1);
    ctx.add(new RamTensor<float>({1}), "MatMul/eightbit/requant_range:1", 1);
    ctx.push(new Requantization_RangeOp(),
             { "MatMul/eightbit:0", "MatMul/eightbit:1", "MatMul/eightbit:2" },
             { "MatMul/eightbit/requant_range:0", "MatMul/eightbit/requant_range:1" });
    ctx.eval();
}
{   
    ctx.add(new RamTensor<uint8_t>(), "MatMul/eightbit/requantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "MatMul/eightbit/requantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "MatMul/eightbit/requantize:2", 1);
    ctx.push(new RequantizeOp(),
             { "MatMul/eightbit:0", "MatMul/eightbit:1", "MatMul/eightbit:2", "MatMul/eightbit/requant_range:0", "MatMul/eightbit/requant_range:1" },
             { "MatMul/eightbit/requantize:0", "MatMul/eightbit/requantize:1", "MatMul/eightbit/requantize:2" });
    ctx.eval();
}
{    
    ctx.add(new BinaryTensor<float>({300}, inline_Variable_1_0), 
            "Variable_1:0", 
            2);
}
{    
    ctx.add(new BinaryTensor<int>({1}, inline_zscore_eightbit_Variable_1__port__0_reshape_dims_0), 
            "zscore_eightbit/Variable_1__port__0/reshape_dims:0", 
            1);
}
{
    ctx.add(new RamTensor<float>(), "zscore_eightbit/Variable_1__port__0/reshape:0", 2);
    ctx.push(new ReshapeOp(), 
             { "Variable_1:0", "zscore_eightbit/Variable_1__port__0/reshape_dims:0" },
             { "zscore_eightbit/Variable_1__port__0/reshape:0" });
    ctx.eval();
}
{    
    ctx.add(new BinaryTensor<int>({1}, inline_zscore_eightbit_Variable_1__port__0_reduction_dims_0), 
            "zscore_eightbit/Variable_1__port__0/reduction_dims:0", 
            2);
}
{   
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "zscore_eightbit/Variable_1__port__0/min:0", 1);
    ctx.push(new MinOp(), 
             { "zscore_eightbit/Variable_1__port__0/reshape:0", "zscore_eightbit/Variable_1__port__0/reduction_dims:0" },
             { "zscore_eightbit/Variable_1__port__0/min:0" });
    ctx.eval();
}
{   
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "zscore_eightbit/Variable_1__port__0/max:0", 1);
    ctx.push(new MaxOp(), 
             { "zscore_eightbit/Variable_1__port__0/reshape:0", "zscore_eightbit/Variable_1__port__0/reduction_dims:0" },
             { "zscore_eightbit/Variable_1__port__0/max:0" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<uint8_t>(), "zscore_eightbit/Variable_1__port__0/quantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "zscore_eightbit/Variable_1__port__0/quantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "zscore_eightbit/Variable_1__port__0/quantize:2", 1);
    ctx.push(new QuantizeV2Op(),
             {  "Variable_1:0",  "zscore_eightbit/Variable_1__port__0/min:0", "zscore_eightbit/Variable_1__port__0/max:0" },
             {  "zscore_eightbit/Variable_1__port__0/quantize:0",  "zscore_eightbit/Variable_1__port__0/quantize:1", "zscore_eightbit/Variable_1__port__0/quantize:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<int>(), "zscore/eightbit:0", 2);
    ctx.add(new RamTensor<float>({1}), "zscore/eightbit:1", 2);
    ctx.add(new RamTensor<float>({1}), "zscore/eightbit:2", 2);
    ctx.push(new QuantizedAddOp<uint8_t, uint8_t, int>(), 
             { "MatMul/eightbit/requantize:0", "MatMul/eightbit/requantize:1", "MatMul/eightbit/requantize:2", "zscore_eightbit/Variable_1__port__0/quantize:0", "zscore_eightbit/Variable_1__port__0/quantize:1",  "zscore_eightbit/Variable_1__port__0/quantize:2" },
             { "zscore/eightbit:0", "zscore/eightbit:1",  "zscore/eightbit:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<float>({1}), "zscore/eightbit/requant_range:0", 1);
    ctx.add(new RamTensor<float>({1}), "zscore/eightbit/requant_range:1", 1);
    ctx.push(new Requantization_RangeOp(),
             { "zscore/eightbit:0", "zscore/eightbit:1", "zscore/eightbit:2" },
             { "zscore/eightbit/requant_range:0", "zscore/eightbit/requant_range:1" });
    ctx.eval();
}
{   
    ctx.add(new RamTensor<uint8_t>(), "zscore/eightbit/requantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "zscore/eightbit/requantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "zscore/eightbit/requantize:2", 1);
    ctx.push(new RequantizeOp(),
             { "zscore/eightbit:0", "zscore/eightbit:1", "zscore/eightbit:2", "zscore/eightbit/requant_range:0", "zscore/eightbit/requant_range:1" },
             { "zscore/eightbit/requantize:0", "zscore/eightbit/requantize:1", "zscore/eightbit/requantize:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<uint8_t>(), "Relu/eightbit:0", 1);
    ctx.add(new RamTensor<float>({1}), "Relu/eightbit:1", 1);
    ctx.add(new RamTensor<float>({1}), "Relu/eightbit:2", 1);
    ctx.push(new ReluOp<uint8_t, float, uint8_t>(), 
             { "zscore/eightbit/requantize:0", "zscore/eightbit/requantize:1", "zscore/eightbit/requantize:2" },
             { "Relu/eightbit:0", "Relu/eightbit:1", "Relu/eightbit:2" });
    ctx.eval();
}
{    
    ctx.add(new BinaryTensor<uint8_t>({300,300}, inline_Variable_2_quantized_const_0), 
            "Variable_2_quantized_const:0", 
            1);
}
{    
    ctx.add(new BinaryTensor<float>({}, inline_Variable_2_quantized_min_0), 
            "Variable_2_quantized_min:0", 
            1);
}
{    
    ctx.add(new BinaryTensor<float>({}, inline_Variable_2_quantized_max_0), 
            "Variable_2_quantized_max:0", 
            1);
}
{
    ctx.add(new RamTensor<int>(), "MatMul_1/eightbit:0", 2);
    ctx.add(new RamTensor<float>({1}), "MatMul_1/eightbit:1", 2);
    ctx.add(new RamTensor<float>({1}), "MatMul_1/eightbit:2", 2);
    ctx.push(new QntMatMulOp<uint8_t, uint8_t, int>(), 
             { "Relu/eightbit:0", "Relu/eightbit:1", "Relu/eightbit:2", "Variable_2_quantized_const:0", "Variable_2_quantized_min:0",  "Variable_2_quantized_max:0" },
             { "MatMul_1/eightbit:0", "MatMul_1/eightbit:1",  "MatMul_1/eightbit:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<float>({1}), "MatMul_1/eightbit/requant_range:0", 1);
    ctx.add(new RamTensor<float>({1}), "MatMul_1/eightbit/requant_range:1", 1);
    ctx.push(new Requantization_RangeOp(),
             { "MatMul_1/eightbit:0", "MatMul_1/eightbit:1", "MatMul_1/eightbit:2" },
             { "MatMul_1/eightbit/requant_range:0", "MatMul_1/eightbit/requant_range:1" });
    ctx.eval();
}
{   
    ctx.add(new RamTensor<uint8_t>(), "MatMul_1/eightbit/requantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "MatMul_1/eightbit/requantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "MatMul_1/eightbit/requantize:2", 1);
    ctx.push(new RequantizeOp(),
             { "MatMul_1/eightbit:0", "MatMul_1/eightbit:1", "MatMul_1/eightbit:2", "MatMul_1/eightbit/requant_range:0", "MatMul_1/eightbit/requant_range:1" },
             { "MatMul_1/eightbit/requantize:0", "MatMul_1/eightbit/requantize:1", "MatMul_1/eightbit/requantize:2" });
    ctx.eval();
}
{    
    ctx.add(new BinaryTensor<float>({300}, inline_Variable_3_0), 
            "Variable_3:0", 
            2);
}
{    
    ctx.add(new BinaryTensor<int>({1}, inline_zscore_1_eightbit_Variable_3__port__0_reshape_dims_0), 
            "zscore_1_eightbit/Variable_3__port__0/reshape_dims:0", 
            1);
}
{
    ctx.add(new RamTensor<float>(), "zscore_1_eightbit/Variable_3__port__0/reshape:0", 2);
    ctx.push(new ReshapeOp(), 
             { "Variable_3:0", "zscore_1_eightbit/Variable_3__port__0/reshape_dims:0" },
             { "zscore_1_eightbit/Variable_3__port__0/reshape:0" });
    ctx.eval();
}
{    
    ctx.add(new BinaryTensor<int>({1}, inline_zscore_1_eightbit_Variable_3__port__0_reduction_dims_0), 
            "zscore_1_eightbit/Variable_3__port__0/reduction_dims:0", 
            2);
}
{   
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "zscore_1_eightbit/Variable_3__port__0/min:0", 1);
    ctx.push(new MinOp(), 
             { "zscore_1_eightbit/Variable_3__port__0/reshape:0", "zscore_1_eightbit/Variable_3__port__0/reduction_dims:0" },
             { "zscore_1_eightbit/Variable_3__port__0/min:0" });
    ctx.eval();
}
{   
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "zscore_1_eightbit/Variable_3__port__0/max:0", 1);
    ctx.push(new MaxOp(), 
             { "zscore_1_eightbit/Variable_3__port__0/reshape:0", "zscore_1_eightbit/Variable_3__port__0/reduction_dims:0" },
             { "zscore_1_eightbit/Variable_3__port__0/max:0" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<uint8_t>(), "zscore_1_eightbit/Variable_3__port__0/quantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "zscore_1_eightbit/Variable_3__port__0/quantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "zscore_1_eightbit/Variable_3__port__0/quantize:2", 1);
    ctx.push(new QuantizeV2Op(),
             {  "Variable_3:0",  "zscore_1_eightbit/Variable_3__port__0/min:0", "zscore_1_eightbit/Variable_3__port__0/max:0" },
             {  "zscore_1_eightbit/Variable_3__port__0/quantize:0",  "zscore_1_eightbit/Variable_3__port__0/quantize:1", "zscore_1_eightbit/Variable_3__port__0/quantize:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<int>(), "zscore_1/eightbit:0", 2);
    ctx.add(new RamTensor<float>({1}), "zscore_1/eightbit:1", 2);
    ctx.add(new RamTensor<float>({1}), "zscore_1/eightbit:2", 2);
    ctx.push(new QuantizedAddOp<uint8_t, uint8_t, int>(), 
             { "MatMul_1/eightbit/requantize:0", "MatMul_1/eightbit/requantize:1", "MatMul_1/eightbit/requantize:2", "zscore_1_eightbit/Variable_3__port__0/quantize:0", "zscore_1_eightbit/Variable_3__port__0/quantize:1",  "zscore_1_eightbit/Variable_3__port__0/quantize:2" },
             { "zscore_1/eightbit:0", "zscore_1/eightbit:1",  "zscore_1/eightbit:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<float>({1}), "zscore_1/eightbit/requant_range:0", 1);
    ctx.add(new RamTensor<float>({1}), "zscore_1/eightbit/requant_range:1", 1);
    ctx.push(new Requantization_RangeOp(),
             { "zscore_1/eightbit:0", "zscore_1/eightbit:1", "zscore_1/eightbit:2" },
             { "zscore_1/eightbit/requant_range:0", "zscore_1/eightbit/requant_range:1" });
    ctx.eval();
}
{   
    ctx.add(new RamTensor<uint8_t>(), "zscore_1/eightbit/requantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "zscore_1/eightbit/requantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "zscore_1/eightbit/requantize:2", 1);
    ctx.push(new RequantizeOp(),
             { "zscore_1/eightbit:0", "zscore_1/eightbit:1", "zscore_1/eightbit:2", "zscore_1/eightbit/requant_range:0", "zscore_1/eightbit/requant_range:1" },
             { "zscore_1/eightbit/requantize:0", "zscore_1/eightbit/requantize:1", "zscore_1/eightbit/requantize:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<uint8_t>(), "Relu_1/eightbit:0", 1);
    ctx.add(new RamTensor<float>({1}), "Relu_1/eightbit:1", 1);
    ctx.add(new RamTensor<float>({1}), "Relu_1/eightbit:2", 1);
    ctx.push(new ReluOp<uint8_t, float, uint8_t>(), 
             { "zscore_1/eightbit/requantize:0", "zscore_1/eightbit/requantize:1", "zscore_1/eightbit/requantize:2" },
             { "Relu_1/eightbit:0", "Relu_1/eightbit:1", "Relu_1/eightbit:2" });
    ctx.eval();
}
{    
    ctx.add(new BinaryTensor<uint8_t>({300,6}, inline_Variable_4_quantized_const_0), 
            "Variable_4_quantized_const:0", 
            1);
}
{    
    ctx.add(new BinaryTensor<float>({}, inline_Variable_4_quantized_min_0), 
            "Variable_4_quantized_min:0", 
            1);
}
{    
    ctx.add(new BinaryTensor<float>({}, inline_Variable_4_quantized_max_0), 
            "Variable_4_quantized_max:0", 
            1);
}
{
    ctx.add(new RamTensor<int>(), "MatMul_2/eightbit:0", 2);
    ctx.add(new RamTensor<float>({1}), "MatMul_2/eightbit:1", 2);
    ctx.add(new RamTensor<float>({1}), "MatMul_2/eightbit:2", 2);
    ctx.push(new QntMatMulOp<uint8_t, uint8_t, int>(), 
             { "Relu_1/eightbit:0", "Relu_1/eightbit:1", "Relu_1/eightbit:2", "Variable_4_quantized_const:0", "Variable_4_quantized_min:0",  "Variable_4_quantized_max:0" },
             { "MatMul_2/eightbit:0", "MatMul_2/eightbit:1",  "MatMul_2/eightbit:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<float>({1}), "MatMul_2/eightbit/requant_range:0", 1);
    ctx.add(new RamTensor<float>({1}), "MatMul_2/eightbit/requant_range:1", 1);
    ctx.push(new Requantization_RangeOp(),
             { "MatMul_2/eightbit:0", "MatMul_2/eightbit:1", "MatMul_2/eightbit:2" },
             { "MatMul_2/eightbit/requant_range:0", "MatMul_2/eightbit/requant_range:1" });
    ctx.eval();
}
{   
    ctx.add(new RamTensor<uint8_t>(), "MatMul_2/eightbit/requantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "MatMul_2/eightbit/requantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "MatMul_2/eightbit/requantize:2", 1);
    ctx.push(new RequantizeOp(),
             { "MatMul_2/eightbit:0", "MatMul_2/eightbit:1", "MatMul_2/eightbit:2", "MatMul_2/eightbit/requant_range:0", "MatMul_2/eightbit/requant_range:1" },
             { "MatMul_2/eightbit/requantize:0", "MatMul_2/eightbit/requantize:1", "MatMul_2/eightbit/requantize:2" });
    ctx.eval();
}
{    
    ctx.add(new BinaryTensor<float>({6}, inline_Variable_5_0), 
            "Variable_5:0", 
            2);
}
{    
    ctx.add(new BinaryTensor<int>({1}, inline_logits_eightbit_Variable_5__port__0_reshape_dims_0), 
            "logits_eightbit/Variable_5__port__0/reshape_dims:0", 
            1);
}
{
    ctx.add(new RamTensor<float>(), "logits_eightbit/Variable_5__port__0/reshape:0", 2);
    ctx.push(new ReshapeOp(), 
             { "Variable_5:0", "logits_eightbit/Variable_5__port__0/reshape_dims:0" },
             { "logits_eightbit/Variable_5__port__0/reshape:0" });
    ctx.eval();
}
{    
    ctx.add(new BinaryTensor<int>({1}, inline_logits_eightbit_Variable_5__port__0_reduction_dims_0), 
            "logits_eightbit/Variable_5__port__0/reduction_dims:0", 
            2);
}
{   
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "logits_eightbit/Variable_5__port__0/min:0", 1);
    ctx.push(new MinOp(), 
             { "logits_eightbit/Variable_5__port__0/reshape:0", "logits_eightbit/Variable_5__port__0/reduction_dims:0" },
             { "logits_eightbit/Variable_5__port__0/min:0" });
    ctx.eval();
}
{   
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "logits_eightbit/Variable_5__port__0/max:0", 1);
    ctx.push(new MaxOp(), 
             { "logits_eightbit/Variable_5__port__0/reshape:0", "logits_eightbit/Variable_5__port__0/reduction_dims:0" },
             { "logits_eightbit/Variable_5__port__0/max:0" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<uint8_t>(), "logits_eightbit/Variable_5__port__0/quantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "logits_eightbit/Variable_5__port__0/quantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "logits_eightbit/Variable_5__port__0/quantize:2", 1);
    ctx.push(new QuantizeV2Op(),
             {  "Variable_5:0",  "logits_eightbit/Variable_5__port__0/min:0", "logits_eightbit/Variable_5__port__0/max:0" },
             {  "logits_eightbit/Variable_5__port__0/quantize:0",  "logits_eightbit/Variable_5__port__0/quantize:1", "logits_eightbit/Variable_5__port__0/quantize:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<int>(), "logits/eightbit:0", 2);
    ctx.add(new RamTensor<float>({1}), "logits/eightbit:1", 2);
    ctx.add(new RamTensor<float>({1}), "logits/eightbit:2", 2);
    ctx.push(new QuantizedAddOp<uint8_t, uint8_t, int>(), 
             { "MatMul_2/eightbit/requantize:0", "MatMul_2/eightbit/requantize:1", "MatMul_2/eightbit/requantize:2", "logits_eightbit/Variable_5__port__0/quantize:0", "logits_eightbit/Variable_5__port__0/quantize:1",  "logits_eightbit/Variable_5__port__0/quantize:2" },
             { "logits/eightbit:0", "logits/eightbit:1",  "logits/eightbit:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<float>({1}), "logits/eightbit/requant_range:0", 1);
    ctx.add(new RamTensor<float>({1}), "logits/eightbit/requant_range:1", 1);
    ctx.push(new Requantization_RangeOp(),
             { "logits/eightbit:0", "logits/eightbit:1", "logits/eightbit:2" },
             { "logits/eightbit/requant_range:0", "logits/eightbit/requant_range:1" });
    ctx.eval();
}
{   
    ctx.add(new RamTensor<uint8_t>(), "logits/eightbit/requantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "logits/eightbit/requantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "logits/eightbit/requantize:2", 1);
    ctx.push(new RequantizeOp(),
             { "logits/eightbit:0", "logits/eightbit:1", "logits/eightbit:2", "logits/eightbit/requant_range:0", "logits/eightbit/requant_range:1" },
             { "logits/eightbit/requantize:0", "logits/eightbit/requantize:1", "logits/eightbit/requantize:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<float>(), "logits:0", 1);
    ctx.push(new DequantizeOp(), 
             { "logits/eightbit/requantize:0", "logits/eightbit/requantize:1", "logits/eightbit/requantize:2" },
             { "logits:0" });
    ctx.eval();
}
{    
    ctx.add(new BinaryTensor<int>({}, inline_y_pred_dimension_0), 
            "y_pred/dimension:0", 
            1);
}
{
    ctx.add(new RamTensor<int>(), "y_pred:0");
    ctx.push(new ArgMaxOp<float, int>(), 
             { "logits:0", "y_pred/dimension:0" },
             { "y_pred:0" });
}
}