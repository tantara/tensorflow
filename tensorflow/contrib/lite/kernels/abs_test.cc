/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class AbsOpModel : public SingleOpModel {
 public:
  explicit AbsOpModel(const TensorData& input) {
    input_ = AddInput(input);
    output_ = AddOutput(input);
    SetBuiltinOp(BuiltinOperator_ABS, BuiltinOptions_AbsOptions,
                 CreateAbsOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }
  int output() { return output_; }

 protected:
  int input_;
  int output_;
};

TEST(AbsOpModel, AbsFloat) {
  AbsOpModel m({TensorType_FLOAT32, {2, 3}});
  m.PopulateTensor<float>(m.input(), {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray({2.0, 1.0, 0.0, 1.0, 2.0, 3.0}));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({2, 3}));
}

TEST(AbsOpModel, AbsInt32) {
  AbsOpModel m({TensorType_INT32, {1, 2, 2, 1}});
  m.PopulateTensor<int32_t>(m.input(), {-2, -1, 0, 3});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<int32_t>(m.output()),
              ElementsAreArray({2, 1, 0, 3}));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 2, 2, 1}));
}

TEST(AbsOpModel, AbsInt64) {
  AbsOpModel m({TensorType_INT64, {1, 2, 2, 1}});
  m.PopulateTensor<int64_t>(m.input(), {-2, -1, 0, 3});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<int64_t>(m.output()),
              ElementsAreArray({2, 1, 0, 3}));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 2, 2, 1}));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}