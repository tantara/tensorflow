/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/contrib/lite/c/c_api_internal.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace abs {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
    
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  output->type = input->type;

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

template <typename T>
void Abs(const T* in_data, int num_elements, T* out_data) {
  // TODO: add vectorized version.
  for (int i = 0; i < num_elements; ++i) {
    out_data[i] = std::abs(in_data[i]);
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const int num_elements = NumElements(input);
  switch (input->type) {
    case kTfLiteInt64:
      Abs(input->data.i64, num_elements, output->data.i64);
      break;
    case kTfLiteInt32:
      Abs(input->data.i32, num_elements, output->data.i32);
      break;
    case kTfLiteFloat32:
      Abs(input->data.f, num_elements, output->data.f);
      break;
    default:
      context->ReportError(context,
                           "Abs only currently supports int64, int32, "
                           "and float32, got %d.",
                           input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace abs 

TfLiteRegistration* Register_ABS() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 abs::Prepare, abs::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite