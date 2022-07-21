/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the GPL v3 License.
* See LICENSE file for more information.
*/

#include "geometry.h"

#include "saiga/vision/torch/ColorizeTensor.h"


std::pair<torch::Tensor, torch::Tensor> HierarchicalNeuralGeometry::AccumulateSampleLossPerNode(
    const NodeBatchedSamples& combined_samples, torch::Tensor per_ray_loss)
{
    // [num_rays]
    per_ray_loss = torch::mean(per_ray_loss, 0).squeeze(0);

    auto linear_ray_index = combined_samples.ray_index.reshape({-1});
    auto per_sample_loss  = per_ray_loss.gather(0, linear_ray_index);
    per_sample_loss       = per_sample_loss.reshape(combined_samples.integration_weight.sizes());

    // [batches, batch_sample_per_node, channels]
    per_sample_loss = per_sample_loss * combined_samples.integration_weight;

    // project sample loss to nodes
    auto per_cell_loss   = torch::sum(per_sample_loss, {1, 2});
    auto per_cell_weight = torch::sum(combined_samples.integration_weight, {1, 2});

    auto per_cell_loss_sum   = torch::zeros({(long)tree->NumNodes()}, device);
    auto per_cell_weight_sum = torch::zeros({(long)tree->NumNodes()}, device);
    per_cell_loss_sum.scatter_add_(0, combined_samples.node_ids, per_cell_loss);
    per_cell_weight_sum.scatter_add_(0, combined_samples.node_ids, per_cell_weight);

    return {per_cell_loss_sum, per_cell_weight_sum};
}


std::pair<torch::Tensor, torch::Tensor> HierarchicalNeuralGeometry::AccumulateSampleLossPerNode(
    const SampleList& combined_samples, torch::Tensor per_ray_loss)
{
    // [num_rays]
    per_ray_loss = torch::mean(per_ray_loss, 0).squeeze(0);

    auto linear_ray_index = combined_samples.ray_index.reshape({-1});
    auto per_sample_loss  = per_ray_loss.gather(0, linear_ray_index);
    per_sample_loss       = per_sample_loss.reshape(combined_samples.weight.sizes());

    // [num_samples]
    per_sample_loss = per_sample_loss * combined_samples.weight;


    auto per_cell_loss_sum   = torch::zeros({(long)tree->NumNodes()}, device);
    auto per_cell_weight_sum = torch::zeros({(long)tree->NumNodes()}, device);
    per_cell_loss_sum.scatter_add_(0, combined_samples.node_id, per_sample_loss);
    per_cell_weight_sum.scatter_add_(0, combined_samples.node_id, combined_samples.weight);

    return {per_cell_loss_sum, per_cell_weight_sum};
}

FCBlock HierarchicalNeuralGeometry::shared_decoder = nullptr;

HierarchicalNeuralGeometry::HierarchicalNeuralGeometry(int num_channels, int D, std::shared_ptr<CombinedParams> params,
                                                       HyperTreeBase tree)
    : NeuralGeometry(num_channels, D, params), tree(tree)
{
    std::cout << ConsoleColor::BLUE << "=== Neural Geometry ===\n";

    int decoder_num_channels = num_channels;

    std::cout << "Last Activation: " << params->net_params.last_activation_function << std::endl;



    if (params->net_params.decoder_skip)
    {
        params->net_params.grid_features = num_channels;
    }
    else
    {
        int features_after_pe = params->net_params.grid_features;


        {
            std::cout << "Decoder\n";
            decoder = FCBlock(features_after_pe, decoder_num_channels, params->net_params.decoder_hidden_layers,
                              params->net_params.decoder_hidden_features, true, params->net_params.decoder_activation);
            register_module("decoder", decoder);
        }

            optimizer_decoder =
                std::make_shared<torch::optim::Adam>(std::vector<torch::Tensor>(), torch::optim::AdamOptions().lr(10));
            std::cout << "Optimizing Decoder with LR " << params->net_params.decoder_lr << std::endl;

            auto options = std::make_unique<torch::optim::AdamOptions>(params->net_params.decoder_lr);
            // options->weight_decay(1e-2);
            // optimizer_adam->add_param_group({decoder->parameters(), std::move(options)});
            optimizer_decoder->add_param_group({decoder->parameters(), std::move(options)});

    }
}

torch::Tensor HierarchicalNeuralGeometry::DecodeFeatures(torch::Tensor neural_features)
{
    //  [num_groups, group_size, num_channels]
    torch::Tensor decoded_features;
    if (decoder)
    {
        CHECK(!shared_decoder);
        decoded_features = decoder->forward(neural_features);
    }
    else if (shared_decoder)
    {
        decoded_features = shared_decoder->forward(neural_features);
    }
    else
    {
        // CHECK(false);
        decoded_features = neural_features;
    }



    if (true)  // params->dataset_params.image_formation == "xray"
    {
        if (params->net_params.last_activation_function == "relu")
        {
            decoded_features = torch::relu(decoded_features);
        }
        else if (params->net_params.last_activation_function == "abs")
        {
            decoded_features = torch::abs(decoded_features);
        }
        else if (params->net_params.last_activation_function == "softplus")
        {
            decoded_features = torch::softplus(decoded_features, params->net_params.softplus_beta);
        }
        else if (params->net_params.last_activation_function == "id")
        {
        }
        else
        {
            CHECK(false);
        }
    }
#if 0
    else if (params->dataset_params.image_formation == "color_density")
    {
        // sigmoid for the color and relu for the density
        int num_channels = decoded_features.size(2);
        CHECK_GT(num_channels, 1);
        auto color       = decoded_features.slice(2, 0, num_channels - 1);
        auto density     = decoded_features.slice(2, num_channels - 1, num_channels);
        color            = torch::sigmoid(color);
        density          = torch::relu(density);
        decoded_features = torch::cat({color, density}, 2);
    }
    else
    {
        CHECK(false);
    }
#endif
    return decoded_features;
}

void HierarchicalNeuralGeometry::AddParametersToOptimizer() {}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> HierarchicalNeuralGeometry::UniformSampledVolume(
    std::vector<long> shape, int num_channels)
{
    Eigen::Vector<int, -1> shape_v;
    shape_v.resize(shape.size());
    for (int i = 0; i < shape.size(); ++i)
    {
        shape_v(i) = shape[i];
    }

    auto vol_shape                      = shape;
    torch::Tensor output_volume_node_id = torch::zeros(vol_shape, torch::kLong);
    torch::Tensor output_volume_valid   = torch::zeros(vol_shape, torch::kLong);

    vol_shape.insert(vol_shape.begin(), num_channels);
    torch::Tensor output_volume = torch::zeros(vol_shape);


    // Generating the whole volume in one batch might cause a memory issue.
    // Therefore, we batch the volume samples into a fixed size.
    // tree->to(torch::kCPU);
    auto all_samples = tree->UniformPhantomSamplesGPU(shape_v, false);
    all_samples.to(device);
    tree->to(device);
    int max_samples = 100000;
    int num_batches = iDivUp(all_samples.size(), max_samples);

    ProgressBar bar(std::cout, "Sampling Volume", num_batches);
    for (int i = 0; i < num_batches; ++i)
    {
        auto samples = all_samples.Slice(i * max_samples, std::min((i + 1) * max_samples, all_samples.size()));

        auto b = tree->GroupSamplesPerNodeGPU(samples, 64);
        b.to(device);
        // auto model_output = pipeline->Forward(b);
        // [num_groups, group_size, num_channels]
        auto model_output = SampleVolumeBatched(b.global_coordinate, b.mask, b.node_ids);

        model_output = model_output * b.integration_weight;


        if (false)  //&&params->dataset_params.image_formation == "color_density"
        {
            model_output =
                model_output.slice(2, 0, num_channels) * model_output.slice(2, num_channels, num_channels + 1);
        }

        // [num_samples, num_channels]
        auto cpu_out = model_output.cpu().contiguous().reshape({-1, num_channels});
        // [num_samples]
        auto cpu_mask   = b.mask.cpu().contiguous().reshape({-1});
        auto cpu_weight = b.integration_weight.cpu().contiguous().reshape({-1});
        // [num_samples]
        auto cpu_ray_index = b.ray_index.cpu().contiguous().reshape({-1});

        // [num_samples]
        auto cpu_node_id = b.node_ids.cpu().contiguous().reshape({-1});



        float* out_ptr       = cpu_out.template data_ptr<float>();
        float* mask_ptr      = cpu_mask.template data_ptr<float>();
        float* weight_ptr    = cpu_weight.template data_ptr<float>();
        long* index_ptr      = cpu_ray_index.template data_ptr<long>();
        long* node_index_ptr = cpu_node_id.template data_ptr<long>();

        long num_samples = cpu_out.size(0);

        for (int i = 0; i < num_samples; ++i)
        {
            if (mask_ptr[i] == 0) continue;
            for (int c = 0; c < num_channels; ++c)
            {
                float col = out_ptr[i * cpu_out.stride(0) + c * cpu_out.stride(1)];
                output_volume.data_ptr<float>()[output_volume.stride(0) * c + index_ptr[i]] = col;
            }
            auto out_node_id                                     = node_index_ptr[i / b.GroupSize()];
            output_volume_node_id.data_ptr<long>()[index_ptr[i]] = out_node_id;
            output_volume_valid.data_ptr<long>()[index_ptr[i]]   = weight_ptr[i];
        }
        bar.addProgress(1);
    }
    return {output_volume, output_volume_node_id, output_volume_valid};
}
void HierarchicalNeuralGeometry::SaveVolume(TensorBoardLogger* tblogger, std::string tb_name, std::string out_dir,
                                            int num_channels, float intensity_scale, int size)
{
    if (size <= 0) return;
    // std::cout << ">> Saving reconstructed volume." << std::endl;
    auto s = size;

    auto [volume_density_raw, volume_node_id, volume_valid] = UniformSampledVolume({s, s, s}, num_channels);

    std::cout << "Volume Raw: " << TensorInfo(volume_density_raw) << std::endl;
    // [channels, z, y, x]
    volume_density_raw = volume_density_raw * intensity_scale;

    // Normalize node error by volume
    auto volume     = torch::prod(tree->node_position_max - tree->node_position_min, {1});
    auto node_error = (tree->node_max_density.clamp_min(0) * tree->node_error.clamp_min(0) / volume).cpu();
    //        auto node_error = (tree->node_error / volume).cpu();

    auto error_volume = torch::index_select(node_error, 0, volume_node_id.reshape({-1}));
    // std::cout << "Volume Max Density:"
    //           << TensorInfo(torch::index_select(tree->node_max_density.cpu(), 0, volume_node_id.reshape({-1})))
    //           << std::endl;
    // std::cout << "Volume Error:"
    //           << TensorInfo(torch::index_select(tree->node_error.cpu(), 0, volume_node_id.reshape({-1}))) <<
    //           std::endl;
    error_volume = error_volume.reshape({s, s, s});
    error_volume = error_volume / error_volume.max().clamp_min(0.01);
    // [3, s, s, s]
    error_volume = ColorizeTensor(error_volume, colorizePlasma);

    // draw inactive nodes in different color
    error_volume                    = error_volume * volume_valid.to(torch::kFloat32);
    std::vector<float> culled_color = {0, 0.7, 0};
    torch::Tensor culled_color_t    = torch::from_blob(culled_color.data(), {3, 1, 1, 1});
    error_volume += (1 - volume_valid.to(torch::kFloat32)) * culled_color_t;

    // [s, 3, s , s]
    error_volume = error_volume.permute({1, 0, 2, 3});

    // set node_id of inactive nodes to -1
    volume_node_id = (volume_node_id * volume_valid) - (1 - volume_valid);

    torch::Tensor volume_density = volume_density_raw;
    if (num_channels == 1)
    {
        // [z, y, x]
        volume_density = volume_density.squeeze(0);
        // [3, z, y, x]
        volume_density = ColorizeTensor(volume_density, colorizeMagma);
        // [z, 3, y, x]
        volume_density = volume_density.permute({1, 0, 2, 3});
    }

    // v = v.clamp(0, 1);
    // v = v / v.max();
    ProgressBar bar(std::cout, "Writing Volume", volume_density.size(0));
    for (int i = 0; i < volume_density.size(0); ++i)
    {
        // [1, 1, h, w]
        auto node_index_img = volume_node_id[i].unsqueeze(0).unsqueeze(0).to(torch::kFloat);
        mat3 sobel_kernel;
        sobel_kernel << 0, 0, 0, -1, 0, 1, 0, 0, 0;
        // [1, h, w]
        auto structure_img_x = Filter2dIndependentChannels(node_index_img, sobel_kernel, 1).squeeze(0);

        mat3 sobel_trans = sobel_kernel.transpose().eval();
        auto structure_img_y = Filter2dIndependentChannels(node_index_img,sobel_trans , 1).squeeze(0);
        auto structure_img   = (structure_img_x.abs() + structure_img_y.abs()).clamp(0, 1);
        // [3, h, w]
        structure_img = structure_img.repeat({3, 1, 1});

        auto error_img = error_volume[i];

        structure_img = (error_img + structure_img).clamp(0, 1);


        auto density_structure = ImageBatchToImageRow(torch::stack({volume_density[i], structure_img}));

        // LogImage(tblogger.get(), TensorToImage<ucvec3>(v[i]), "reconstruction_" +
        // std::to_string(epoch_id),
        //          i);
        //
        // LogImage(tblogger.get(), TensorToImage<ucvec3>(structure_img),
        //         "structure_" + std::to_string(epoch_id), i);

        auto saiga_img = TensorToImage<ucvec3>(density_structure);

        if (!out_dir.empty())
        {
            TensorToImage<ucvec3>(structure_img).save(out_dir + "/" + leadingZeroString(i, 5) + "_structure.png");
            TensorToImage<unsigned char>(volume_density_raw.slice(1, i, i + 1).squeeze(1))
                .save(out_dir + "/" + leadingZeroString(i, 5) + "_density.png");
        }
        LogImage(tblogger, saiga_img, tb_name, i);
        bar.addProgress(1);
    }
}
torch::Tensor HierarchicalNeuralGeometry::ComputeImage(SampleList all_samples, int num_channels, int num_pixels)
{
    torch::Tensor sample_output, weight, ray_index;


    // try direct sampling. if it is not implemtend, an undefined tensor is returned
    sample_output = SampleVolumeIndirect(all_samples.global_coordinate, all_samples.node_id);

    if (sample_output.defined())
    {
        weight    = all_samples.weight.unsqueeze(1);
        ray_index = all_samples.ray_index;
    }
    else
    {
        // use indirect bachted sampling
        NodeBatchedSamples image_samples;

        {
            SAIGA_OPTIONAL_TIME_MEASURE("GroupSamplesPerNodeGPU", timer);
            image_samples = tree->GroupSamplesPerNodeGPU(all_samples, params->train_params.per_node_batch_size);
        }

        // [num_groups, group_size, num_channels]
        sample_output =
            SampleVolumeBatched(image_samples.global_coordinate, image_samples.mask, image_samples.node_ids);

        weight    = image_samples.integration_weight;
        ray_index = image_samples.ray_index;
    }



    torch::Tensor predicted_image;
    {
        SAIGA_OPTIONAL_TIME_MEASURE("IntegrateSamplesXRay", timer);
        // [num_channels, num_rays]
        predicted_image = IntegrateSamplesXRay(sample_output, weight, ray_index, num_channels, num_pixels);
        CHECK_EQ(predicted_image.size(0), num_channels);
    }

    return predicted_image;
}



torch::Tensor NeuralGeometry::IntegrateSamplesXRay(torch::Tensor sample_values, torch::Tensor integration_weight,
                                                   torch::Tensor ray_index, int num_channels, int num_rays)
{
    // [num_channels, num_rays]
    auto density_integral = torch::zeros({num_channels, num_rays}, torch::TensorOptions(ray_index.device()));

    if (sample_values.numel() == 0)
    {
        return density_integral;
    }


    CHECK_EQ(sample_values.dim(), integration_weight.dim());
    sample_values = sample_values * integration_weight;

    // [num_channels, num_samples]
    auto linear_sample_output = sample_values.reshape({-1, num_channels}).permute({1, 0});

    // [num_samples]
    auto linear_ray_index = ray_index.reshape({-1});


    // PrintTensorInfo(density_integral);
    // PrintTensorInfo(linear_ray_index);
    // PrintTensorInfo(linear_sample_output);
    density_integral.index_add_(1, linear_ray_index, linear_sample_output);


    CHECK_EQ(density_integral.dim(), 2);

    return density_integral;
}


torch::Tensor NeuralGeometry::IntegrateSamplesAlphaBlending(torch::Tensor sample_values,
                                                            torch::Tensor integration_weight, torch::Tensor ray_index,
                                                            torch::Tensor sample_index_in_ray, int num_channels,
                                                            int num_rays, int max_samples_per_ray)
{
    // CHECK(params->dataset_params.image_formation == "color_density");
    if (sample_values.numel() == 0)
    {
        return torch::zeros({num_channels - 1, num_rays}, sample_values.device());
    }
    CHECK_GT(num_rays, 0);
    CHECK_GT(max_samples_per_ray, 0);
    CHECK_GT(num_channels, 0);
    CHECK(sample_index_in_ray.defined());

    //    std::cout << "Test IntegrateSamplesAlphaBlending " << num_rays << " " << max_samples_per_ray << std::endl;

    // Linearize all samples in one long array
    sample_values       = sample_values.reshape({-1, num_channels});
    ray_index           = ray_index.reshape({-1});
    sample_index_in_ray = sample_index_in_ray.reshape({-1});
    integration_weight  = integration_weight.reshape({-1, 1});

    // Multiply density by integration weight (the other channels are not modified)
    //    PrintTensorInfo(sample_values);
    //    PrintTensorInfo(integration_weight);
    sample_values.slice(1, num_channels - 1, num_channels) *= integration_weight;



    // [num_rays, max_samples_per_ray, num_channels]
    torch::Tensor sample_matrix  = torch::zeros({num_rays, max_samples_per_ray, num_channels},
                                                torch::TensorOptions(torch::kFloat).device(sample_values.device()));
    auto sample_offset_in_matrix = ray_index * max_samples_per_ray + sample_index_in_ray;
    auto tmp      = sample_matrix.reshape({-1, num_channels}).index_add_(0, sample_offset_in_matrix, sample_values);
    sample_matrix = tmp.reshape(sample_matrix.sizes());


    if (0)
    {
        // sum over the sample dimension
        // [num_channels, num_rays]
        auto simple_integral = sample_matrix.sum({1}).permute({1, 0});
        return simple_integral;
    }


    // We use the last channel as the density
    // [num_rays, num_samples_per_ray]
    auto density = sample_matrix.slice(2, num_channels - 1, num_channels).squeeze(2);
    //    PrintTensorInfo(density);

    // std::cout << density.slice(0,0,1).squeeze(0) << std::endl;

    // density to alpha
    // raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    // [num_rays, num_samples_per_ray]
    auto alpha = 1. - torch::exp(-density * 10);

    // std::cout << alpha[0] << std::endl;
    // PrintTensorInfo(alpha);
    // std::cout << alpha.slice(0,0,1).squeeze(0) << std::endl;

    // Exclusive prefix product on '1-alpha'
    //  weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    auto expanded_by_one     = torch::cat({torch::ones({num_rays, 1}, alpha.options()), 1. - alpha + 1e-10}, 1);
    auto prod                = torch::cumprod(expanded_by_one, 1);
    auto original_shape_prod = prod.slice(1, 0, max_samples_per_ray);
    CHECK_EQ(original_shape_prod.sizes(), alpha.sizes());
    // [num_rays, max_samples_per_ray]
    auto weights = alpha * original_shape_prod;
    // PrintTensorInfo(weights);
    // std::cout << density[0].slice(0, 0, 10) << std::endl;
    // std::cout << alpha[0].slice(0, 0, 10) << std::endl;
    // std::cout << weights[0].slice(0, 0, 10) << std::endl;
    // std::cout << original_shape_prod[0] << std::endl;
    // exit(0);

    // The remaining values are the color
    // [num_rays, max_samples_per_ray, num_color_channels]
    auto rgb = sample_matrix.slice(2, 0, num_channels - 1);

    //    std::cout << "density" << std::endl;
    //    PrintTensorInfo(density);
    //
    //    std::cout << "rgb" << std::endl;
    //    PrintTensorInfo(rgb.slice(2, 0, 1));
    //    PrintTensorInfo(rgb.slice(2, 1, 2));
    //    PrintTensorInfo(rgb.slice(2, 2, 3));
    //    PrintTensorInfo(rgb);

    // weighted sum over density values
    // [num_rays, num_color_channels]
    auto rgb_map = torch::sum(weights.unsqueeze(2) * rgb, {1});
    //    PrintTensorInfo(rgb_map);



    return rgb_map.permute({1, 0});
}
