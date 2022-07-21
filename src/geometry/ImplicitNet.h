/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the GPL v3 License.
* See LICENSE file for more information.
*/

#pragma once

#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"
#include "saiga/vision/torch/TorchHelper.h"

#include <torch/torch.h>



class FCBlockImpl : public torch::nn::Module
{
   public:
    FCBlockImpl(int in_features, int out_features, int num_hidden_layers, float hidden_features,
                bool outermost_linear = true, std::string non_linearity = "relu")
        : in_features(in_features), out_features(out_features)
    {
        auto make_lin = [](int in, int out)
        {
            auto lin = torch::nn::Linear(in, out);
            torch::nn::init::kaiming_normal_(lin->weight, 0, torch::kFanIn, torch::kReLU);
            std::cout << "(lin " << in << "->" << out << ") ";
            return lin;
        };

        seq->push_back(make_lin(in_features, hidden_features));
        seq->push_back(Saiga::ActivationFromString(non_linearity));
        std::cout << "(" << non_linearity << ") ";

        for (int i = 0; i < num_hidden_layers; ++i)
        {
            seq->push_back(make_lin(hidden_features, hidden_features));
            seq->push_back(Saiga::ActivationFromString(non_linearity));
            std::cout << "(" << non_linearity << ") ";
        }

        seq->push_back(make_lin(hidden_features, out_features));
        if (!outermost_linear)
        {
            seq->push_back(Saiga::ActivationFromString(non_linearity));
            std::cout << "(" << non_linearity << ") ";
        }
        register_module("seq", seq);

        int num_params = 0;
        for (auto& t : this->parameters())
        {
            num_params += t.numel();
        }
        std::cout << "  |  #Params " << num_params;
        std::cout << std::endl;
    }

    at::Tensor forward(at::Tensor x)
    {
        CHECK_EQ(in_features, x.size(-1));
        x = seq->forward(x);
        CHECK_EQ(out_features, x.size(-1));
        return x;
    }

    int in_features, out_features;
    torch::nn::Sequential seq;
};

TORCH_MODULE(FCBlock);


// Stores a feature grid for every node explicitly in memory.
// The forward function then copies the feature grid into an output tensor based on the
// the node index (index_select).
class ExplicitFeatureGridImpl : public torch::nn::Module
{
   public:
    ExplicitFeatureGridImpl(int num_nodes, int out_features, std::vector<long> output_grid, std::string init = "zero")
    {
        std::vector<long> sizes;
        sizes.push_back(num_nodes);
        sizes.push_back(out_features);
        for (auto g : output_grid)
        {
            sizes.push_back(g);
        }

        if (init == "uniform" || init == "random")
        {
            grid_data = torch::empty(sizes);
            grid_data.uniform_(-1, 1);
        }
        else if (init == "minus")
        {
            grid_data = -torch::ones(sizes);
        }
        else if (init == "zero")
        {
            grid_data = torch::zeros(sizes);
        }
        else
        {
            CHECK(false) << "Unknown grid init: " << init << ". Expected: uniform, minus, zero";
        }
        register_parameter("grid_data", grid_data);
    }

    at::Tensor forward(at::Tensor node_index)
    {
        auto feature_grid = torch::index_select(grid_data, 0, node_index);
        return feature_grid;
    }

    torch::Tensor grid_data;
};
TORCH_MODULE(ExplicitFeatureGrid);


class NeuralGridSamplerImpl : public torch::nn::Module
{
   public:
    NeuralGridSamplerImpl(bool swap_xy, bool align_corners) : swap_xy(swap_xy), align_corners(align_corners) {}

    torch::Tensor forward(at::Tensor features_in, at::Tensor relative_coordinates)
    {
        CHECK_EQ(relative_coordinates.dim(), 3);
        int D = relative_coordinates.size(2);
        CHECK_EQ(features_in.dim(), D + 2);

        auto opt = torch::nn::functional::GridSampleFuncOptions();
        opt      = opt.padding_mode(torch::kBorder).mode(torch::kBilinear).align_corners(align_corners);

        // Note: grid_sample has xy indexing. The tree and everything has yx indexing.
        // -> swap coordinates for grid_sample
        if (D == 2)
        {
            if (swap_xy)
            {
                relative_coordinates =
                    torch::cat({relative_coordinates.slice(2, 1, 2), relative_coordinates.slice(2, 0, 1)}, 2);
            }
            relative_coordinates = relative_coordinates.unsqueeze(1);
        }
        else if (D == 3)
        {
            if (swap_xy)
            {
                relative_coordinates =
                    torch::cat({relative_coordinates.slice(2, 2, 3), relative_coordinates.slice(2, 1, 2),
                                relative_coordinates.slice(2, 0, 1)},
                               2);
            }
            relative_coordinates = relative_coordinates.unsqueeze(1).unsqueeze(1);
        }
        else
        {
            CHECK(false);
        }
        // 3D: [batches, num_features, 1, 1, batch_size]
        auto neural_samples = torch::nn::functional::grid_sample(features_in, relative_coordinates, opt);

        // After squeeze:
        // [batches, num_features, batch_size]
        if (D == 2)
        {
            neural_samples = neural_samples.squeeze(2);
        }
        else if (D == 3)
        {
            neural_samples = neural_samples.squeeze(2).squeeze(2);
        }
        else
        {
            CHECK(false);
        }

        // [batches, batch_size, num_features]
        neural_samples = neural_samples.permute({0, 2, 1});

        return neural_samples;
    }

    bool swap_xy;
    bool align_corners;
};
TORCH_MODULE(NeuralGridSampler);
