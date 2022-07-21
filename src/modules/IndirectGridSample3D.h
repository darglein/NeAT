/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the GPL v3 License.
* See LICENSE file for more information.
*/

#pragma once
#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/DataStructures/ArrayView.h"
#include "saiga/core/util/assert.h"
#include "saiga/vision/VisionTypes.h"

#include <torch/torch.h>


namespace torch::autograd
{
struct IndirectGridSample3D : public Function<IndirectGridSample3D>
{
    // returns a tensor for every layer
    static std::vector<torch::Tensor> forward(AutogradContext* ctx, torch::Tensor multi_grid, torch::Tensor index, torch::Tensor uv);

    static std::vector<torch::Tensor> backward(AutogradContext* ctx, std::vector<torch::Tensor> grad_output);
};
}  // namespace torch::autograd

// Input:
//      multi_grid: [nodes, channels, z, y, x]
torch::Tensor IndirectGridSample3D(torch::Tensor multi_grid, torch::Tensor index, torch::Tensor uv);
