/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the GPL v3 License.
* See LICENSE file for more information.
*/

#pragma once
#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"

#include "ImplicitNet.h"
#include "Settings.h"
#include "data/SceneBase.h"
#include "geometry.h"


class TVLoss
{
   public:
    // TV Loss for N-dimensional input.
    //
    // Input
    //      grid [num_batches, num_channels, x, y, z, ...]
    //      weight [num_batches]
    torch::Tensor forward(torch::Tensor grid, torch::Tensor weight = {})
    {
        int num_channels = grid.size(1);
        int D            = grid.dim() - 2;
        // int num_batches  = grid.size(0);
        // int num_channels = grid.size(1);


        torch::Tensor total_loss;
        for (int i = 0; i < D; ++i)
        {
            int d     = i + 2;
            int size  = grid.size(d);
            auto loss = (grid.slice(d, 0, size - 1) - grid.slice(d, 1, size)).abs().mean({1, 2, 3, 4});

            if (weight.defined())
            {
                loss *= weight;
            }

            loss = loss.mean();

            if (total_loss.defined())
            {
                total_loss += loss;
            }
            else
            {
                total_loss = loss;
            }
        }
        CHECK(total_loss.defined());

        return total_loss * num_channels;
    }
};


class GeometryExEx : public HierarchicalNeuralGeometry
{
   public:
    GeometryExEx(int num_channels, int D, HyperTreeBase tree, std::shared_ptr<CombinedParams> params);


    torch::Tensor VolumeRegularizer();

    virtual void InterpolateInactiveNodes(HyperTreeBase old_tree);

   protected:
    void AddParametersToOptimizer() override;

    torch::Tensor SampleVolume(torch::Tensor global_coordinate, torch::Tensor node_id) override;

    virtual torch::Tensor SampleVolumeIndirect(torch::Tensor global_coordinate, torch::Tensor node_id) override;


    ExplicitFeatureGrid explicit_grid_generator = nullptr;
    NeuralGridSampler grid_sampler              = nullptr;
};
