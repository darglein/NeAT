/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the GPL v3 License.
* See LICENSE file for more information.
*/
#include "saiga/core/image/freeimage.h"
#include "saiga/core/math/random.h"
#include "saiga/core/time/time.h"
#include "saiga/core/util/ConsoleColor.h"
#include "saiga/core/util/ProgressBar.h"
#include "saiga/vision/torch/ColorizeTensor.h"
#include "saiga/vision/torch/ImageSimilarity.h"
#include "saiga/vision/torch/ImageTensor.h"
#include "saiga/vision/torch/TrainParameters.h"

#include "Settings.h"
#include "data/Dataloader.h"
#include "utils/utils.h"

#include "build_config.h"
#include "geometry/geometry_ex_ex.h"
#include "tensorboard_logger.h"
#include "utils/cimg_wrapper.h"
using namespace Saiga;

struct TrainScene
{
    std::shared_ptr<SceneBase> scene;
    HyperTreeBase tree = nullptr;
    std::shared_ptr<HierarchicalNeuralGeometry> neural_geometry;

    double last_eval_loss           = 9237643867809436;
    double new_eval_loss            = 9237643867809436;
    int last_structure_change_epoch = 0;

    void SaveCheckpoint(const std::string& dir)
    {
        auto prefix = dir + "/" + scene->scene_name + "_";
        scene->SaveCheckpoint(dir);

        torch::nn::ModuleHolder<HierarchicalNeuralGeometry> holder(neural_geometry);
        torch::save(holder, prefix + "geometry.pth");

        torch::save(tree, prefix + "tree.pth");
    }

    void LoadCheckpoint(const std::string& dir)
    {
        auto prefix = dir + "/" + scene->scene_name + "_";
        scene->LoadCheckpoint(dir);

        if (std::filesystem::exists(prefix + "geometry.pth"))
        {
            std::cout << "Load checkpoint geometry " << std::endl;
            torch::nn::ModuleHolder<HierarchicalNeuralGeometry> holder(neural_geometry);
            torch::load(holder, prefix + "geometry.pth");
        }
        if (std::filesystem::exists(prefix + "tree.pth"))
        {
            torch::load(tree, prefix + "tree.pth");
        }
    }
};

class Trainer
{
   public:
    Trainer(std::shared_ptr<CombinedParams> params, std::string experiment_dir)
        : params(params), experiment_dir(experiment_dir)
    {
        torch::set_num_threads(4);
        torch::manual_seed(params->train_params.random_seed);

        tblogger = std::make_shared<TensorBoardLogger>((experiment_dir + "/tfevents.pb").c_str());

        for (auto scene_name : params->train_params.scene_name)
        {
            auto scene = std::make_shared<SceneBase>(params->train_params.scene_dir + "/" + scene_name);
            scene->train_indices =
                TrainParams::ReadIndexFile(scene->scene_path + "/" + params->train_params.split_name + "/train.txt");
            scene->test_indices =
                TrainParams::ReadIndexFile(scene->scene_path + "/" + params->train_params.split_name + "/eval.txt");

            if (params->train_params.train_on_eval)
            {
                std::cout << "Train on eval!" << std::endl;
                scene->test_indices = scene->train_indices;
            }

            std::cout << "Train(" << scene->train_indices.size() << "): " << array_to_string(scene->train_indices, ' ')
                      << std::endl;
            std::cout << "Test(" << scene->test_indices.size() << "): " << array_to_string(scene->test_indices, ' ')
                      << std::endl;

            scene->params = params;
            scene->LoadImagesCT(scene->train_indices);
            scene->LoadImagesCT(scene->test_indices);
            scene->Finalize();

            if (params->train_params.init_bias_with_bg)
            {
                scene->InitializeBiasWithBackground(tblogger.get());
            }
            scene->Draw(tblogger.get());

            auto prefix = params->train_params.checkpoint_directory + "/" + scene->scene_name + "_";
            auto tree   = HyperTreeBase(3, params->octree_params.tree_depth);
            tree->SetActive(params->octree_params.start_layer);

            TrainScene ts;
            ts.neural_geometry = std::make_shared<GeometryExEx>(scene->num_channels, scene->D, tree, params);

            ts.scene = scene;
            ts.tree  = tree;
            ts.LoadCheckpoint(params->train_params.checkpoint_directory);
            ts.tree->to(device);
            ts.neural_geometry->to(device);
            scenes.push_back(ts);
        }
    }

    void Train()
    {
        for (int epoch_id = 0; epoch_id <= params->train_params.num_epochs; ++epoch_id)
        {
            bool checkpoint_it = epoch_id % params->train_params.save_checkpoints_its == 0 ||
                                 epoch_id == params->train_params.num_epochs;
            std::string ep_str = Saiga::leadingZeroString(epoch_id, 4);
            std::string cp_str = checkpoint_it ? "Checkpoint(" + ep_str + ")" : "";
            std::cout << "\n==== Epoch " << epoch_id << " ====" << std::endl;
            if (epoch_id > 0)
            {
                TrainStep(epoch_id, true, "Train", false);
            }

            for (auto& ts : scenes)
            {
                auto scene           = ts.scene;
                auto tree            = ts.tree;
                auto neural_geometry = ts.neural_geometry;
                ComputeMaxDensity(ts);
                ts.last_eval_loss = ts.new_eval_loss;
                ts.new_eval_loss =
                    EvalStepProjection(ts, scene->train_indices, "Eval/" + scene->scene_name, epoch_id, cp_str, false);
                if (!params->train_params.train_on_eval)
                {
                    EvalStepProjection(ts, scene->test_indices, "Test/" + scene->scene_name, epoch_id, cp_str, true);
                }
                neural_geometry->PrintInfo();
                scene->PrintInfo(epoch_id, tblogger.get());
            }

            if (checkpoint_it)
            {
                auto ep_dir = experiment_dir + "ep" + ep_str + "/";
                std::cout << "Saving Checkpoint to " << ep_dir << std::endl;
                std::filesystem::create_directory(ep_dir);

                for (auto& ts : scenes)
                {
                    auto scene           = ts.scene;
                    auto tree            = ts.tree;
                    auto neural_geometry = ts.neural_geometry;
                    if (params->train_params.output_volume_size > 0)
                    {
                        auto volume_out_dir = ep_dir + "/volume_" + scene->scene_name + "/";
                        std::filesystem::create_directories(volume_out_dir);

                        int out_size = params->train_params.output_volume_size;
                        if (epoch_id == params->train_params.num_epochs)
                        {
                            // double resolution in last epoch
                            out_size *= 2;

                            // save last epoch as hdr image as well
                            std::cout << "saving volume as .hdr..." << std::endl;
                            auto [volume_density, volume_node_id, volume_valid] = neural_geometry->UniformSampledVolume(
                                {out_size, out_size, out_size}, scene->num_channels);
                            SaveHDRImageTensor(volume_density, volume_out_dir + "/volume.hdr");
                        }
                        else
                        {
                            volume_out_dir = "";
                        }

                        neural_geometry->SaveVolume(tblogger.get(), cp_str + "/volume" + "/" + scene->scene_name,
                                                    volume_out_dir, scene->num_channels,
                                                    scene->dataset_params.vis_volume_intensity_factor, out_size);
                    }

                    ts.SaveCheckpoint(ep_dir);

                    if (scene->ground_truth_volume.defined())
                    {
                        EvalVolume(ts, cp_str + "/volume_gt" + "/" + scene->scene_name, epoch_id,
                                   epoch_id == params->train_params.num_epochs, ep_dir);
                    }
                }
            }

            // don't split before first epoch and after last epoch
            if (params->octree_params.optimize_structure &&
                epoch_id > params->train_params.optimize_tree_structure_after_epochs &&
                epoch_id < params->train_params.num_epochs)
            {
                for (auto& ts : scenes)
                {
                    if (epoch_id <
                        ts.last_structure_change_epoch + params->train_params.optimize_structure_every_epochs)
                    {
                        // has recently updated the structure
                        continue;
                    }

                    float converged_threshold = params->train_params.optimize_structure_convergence;
                    if (ts.new_eval_loss < ts.last_eval_loss * converged_threshold)
                    {
                        // not converged enough yet -> don't do structure change
                        std::cout << "Skip Structure Opt. (not converged) " << (float)ts.last_eval_loss << " -> "
                                  << (float)ts.new_eval_loss << " | " << float(ts.new_eval_loss / ts.last_eval_loss)
                                  << "<" << converged_threshold << std::endl;
                        continue;
                    }

                    auto scene           = ts.scene;
                    auto neural_geometry = ts.neural_geometry;


                    // torch::save(tree, "/tmp/tree.pth");
                    HyperTreeBase old_tree = HyperTreeBase(3, params->octree_params.tree_depth);
                    ts.tree->CloneInto(old_tree.get());
                    // torch::load(old_tree, "/tmp/tree.pth");


                    if (params->octree_params.node_culling)
                    {
                        NodeCulling(ts);
                    }

                    // clone old tree structure

                    OptimizeTreeStructure(ts, epoch_id);
                    neural_geometry->InterpolateInactiveNodes(old_tree);

                    ts.last_structure_change_epoch = epoch_id;
                    ts.new_eval_loss               = 93457345345;
                    ts.last_eval_loss              = 93457345345;
                }
            }
        }
    }

   private:
    void ComputeMaxDensity(TrainScene& ts)
    {
        std::cout << "ComputeMaxDensity" << std::endl;
        auto scene    = ts.scene;
        auto tree     = ts.tree;
        auto geometry = ts.neural_geometry;

        torch::NoGradGuard ngg;

        auto active_node_id = tree->ActiveNodeTensor();

        // Create for each node a 16^3 cube of samples
        // [num_nodes, 16, 16, 16, 3]
        auto node_grid_position = tree->UniformGlobalSamples(active_node_id, params->net_params.grid_size);
        int num_nodes           = node_grid_position.size(0);
        // [num_nodes, group_size, 3]
        node_grid_position = node_grid_position.reshape({num_nodes, -1, 3});

        //        auto node_grid_mask = torch::ones({num_nodes, group_size, 1},
        //        node_grid_position.options());
        auto node_grid_mask = scene->PointInAnyImage(node_grid_position).unsqueeze(2);

        // [num_nodes, group_size, 1]
        auto density = geometry->SampleVolumeBatched(node_grid_position, node_grid_mask, active_node_id);

        // [num_nodes]
        auto [per_node_max_density, max_index] = density.reshape({density.size(0), -1}).max(1);

        auto tree_max_density = torch::zeros_like(tree->node_max_density);
        // tree_max_density.scatter_add_(0, tree->active_node_ids,
        // per_node_max_density);

        tree->node_max_density.index_copy_(0, tree->active_node_ids, per_node_max_density);
    }

    void NodeCulling(TrainScene& ts)
    {
        {
            auto scene    = ts.scene;
            auto tree     = ts.tree;
            auto geometry = ts.neural_geometry;

            torch::NoGradGuard ngg;

            auto new_culled_nodes =
                ((tree->node_max_density < params->octree_params.culling_threshold) * tree->node_active)
                    .to(torch::kInt32);

            int num_culled_nodes = new_culled_nodes.sum().item().toInt();
            std::cout << "Culling " << num_culled_nodes << " nodes" << std::endl;

            if (num_culled_nodes > 0)
            {
                // Integrate new culling into the tree
                tree->node_culled = tree->node_culled.to(device);
                tree->node_culled.add_(new_culled_nodes).clamp_(0, 1);
                tree->node_active.add_(new_culled_nodes, -1).clamp_(0, 1);
                tree->UpdateActive();
                tree->UpdateCulling();

                std::cout << "Resetting Optimizer..." << std::endl;
                geometry->ResetGeometryOptimizer();
                current_lr_factor = 1;
            }
        }
    }

    void TrainStep(int epoch_id, bool train_indices, std::string name, bool only_image_params)
    {
        std::vector<std::vector<int>> indices_list;
        std::vector<std::shared_ptr<SceneBase>> scene_list;

        for (auto& ts : scenes)
        {
            auto scene           = ts.scene;
            auto tree            = ts.tree;
            auto neural_geometry = ts.neural_geometry;
            auto indices         = train_indices ? scene->train_indices : scene->test_indices;

            indices_list.push_back(indices);
            scene_list.push_back(scene);

            neural_geometry->train(epoch_id, true);
            scene->train(true);
        }

        auto options = torch::data::DataLoaderOptions()
                           .batch_size(params->train_params.batch_size)
                           .drop_last(false)
                           .workers(params->train_params.num_workers_train);
        auto dataset     = RandomMultiSceneDataset(indices_list, scene_list, params,
                                                   params->train_params.rays_per_image * (only_image_params ? 0.25 : 1));
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(dataset, options);

        // optimize network with fixed structure
        Saiga::ProgressBar bar(std::cout, name + " " + std::to_string(epoch_id) + " |", dataset.size().value(), 30,
                               false, 1000, "ray");

        float epoch_loss_train  = 0;
        int processed_ray_count = 0;
        std::vector<double> batch_loss;

        for (SampleData sample_data : (*data_loader))
        {
            auto& scene           = scenes[sample_data.scene_id].scene;
            auto& tree            = scenes[sample_data.scene_id].tree;
            auto& neural_geometry = scenes[sample_data.scene_id].neural_geometry;

            RayList rays =
                scene->GetRays(sample_data.pixels.uv, sample_data.pixels.image_id, sample_data.pixels.camera_id);
            SampleList all_samples = tree->CreateSamplesForRays(rays, params->octree_params.max_samples_per_node, true);

            auto predicted_image =
                neural_geometry->ComputeImage(all_samples, scene->num_channels, sample_data.NumPixels());
            predicted_image =
                scene->tone_mapper->forward(predicted_image, sample_data.pixels.uv, sample_data.pixels.image_id);
            if (sample_data.pixels.target_mask.defined())
            {
                // Multiply by mask so the loss of invalid pixels is 0
                predicted_image           = predicted_image * sample_data.pixels.target_mask;
                sample_data.pixels.target = sample_data.pixels.target * sample_data.pixels.target_mask;
            }

            CHECK_EQ(predicted_image.sizes(), sample_data.pixels.target.sizes());
            auto per_ray_loss_mse = ((predicted_image - sample_data.pixels.target)).square();
            auto per_ray_loss     = per_ray_loss_mse;

            auto avg_per_image_loss = (per_ray_loss).mean();

            auto volume_loss = neural_geometry->VolumeRegularizer();
            auto param_loss  = scene->tone_mapper->ParameterLoss(scene->active_train_images);
            // auto total_loss  = avg_per_image_loss + param_loss;
            auto total_loss = avg_per_image_loss + param_loss;
            if (volume_loss.defined())
            {
                total_loss += volume_loss;
            }

            static int global_batch_id_a = 0;
            static int global_batch_id_b = 0;
            int& global_batch_id         = only_image_params ? global_batch_id_b : global_batch_id_a;

            total_loss.backward();
            if (!only_image_params)
            {
                neural_geometry->PrintGradInfo(global_batch_id, tblogger.get());
                neural_geometry->OptimizerStep(epoch_id);
            }
            scene->PrintGradInfo(global_batch_id, tblogger.get());
            scene->OptimizerStep(epoch_id, only_image_params);

            float avg_per_image_loss_float = avg_per_image_loss.item().toFloat();
            tblogger->add_scalar("Loss" + name + "/" + scene->scene_name + "/batch", global_batch_id,
                                 avg_per_image_loss_float);
            batch_loss.push_back(avg_per_image_loss_float);
            // tblogger->add_scalar("TrainLoss/param", global_batch_id,
            // param_loss.item().toFloat());
            epoch_loss_train += avg_per_image_loss_float * sample_data.NumPixels();
            processed_ray_count += sample_data.NumPixels();

            // float param_loss_float = param_loss.item().toFloat();
            float regularizer_loss = 0;
            if (volume_loss.defined())
            {
                regularizer_loss = volume_loss.item().toFloat();
            }

            bar.SetPostfix(" Cur=" + std::to_string(epoch_loss_train / processed_ray_count) +
                           // " Param: " + std::to_string(param_loss_float) +
                           " Reg: " + std::to_string(regularizer_loss));
            bar.addProgress(sample_data.NumPixels());

            global_batch_id++;
        }

        if (!only_image_params)
        {
            std::ofstream strm(experiment_dir + "/batch_loss.txt", std::ios_base::app);
            for (auto d : batch_loss)
            {
                strm << d << "\n";
            }
        }

        tblogger->add_scalar("TrainLoss/lr_factor", epoch_id, current_lr_factor);
        current_lr_factor *= params->train_params.lr_decay_factor;

        for (auto& ts : scenes)
        {
            auto scene           = ts.scene;
            auto tree            = ts.tree;
            auto neural_geometry = ts.neural_geometry;
            neural_geometry->UpdateLearningRate(params->train_params.lr_decay_factor);
            scene->UpdateLearningRate(params->train_params.lr_decay_factor);
        }
        tblogger->add_scalar("Loss" + name + "/total", epoch_id, epoch_loss_train / processed_ray_count);
    }

    void EvalVolume(TrainScene& ts, std::string tbname, int epoch_id, bool write_gt, std::string epoch_dir)
    {
        std::string name = ts.scene->scene_name;
        auto target      = ts.scene->ground_truth_volume;

        auto [volume, volume_node_id, volume_valid] = ts.neural_geometry->UniformSampledVolume(
            {target.size(1), target.size(2), target.size(3)}, ts.scene->num_channels);

        std::cout << "gt:  " << TensorInfo(target) << std::endl;
        std::cout << "rec: " << TensorInfo(volume) << std::endl;

        PSNR psnr(0, 5);
        auto volume_error_psnr = psnr->forward(volume, target);

        SSIM3D ssim(5, 1);
        auto volume_ssim = ssim->forward(volume.unsqueeze(0), target.unsqueeze(0));

        float epoch_loss_train_psnr = volume_error_psnr.item().toFloat();
        float epoch_loss_train_ssim = volume_ssim.item().toFloat();

        {
            std::ofstream loss_file(experiment_dir + "/psnr_ssim.txt", std::ios_base::app);
            loss_file << epoch_loss_train_psnr << "," << epoch_loss_train_ssim << "\n";
        }

        tblogger->add_scalar("Loss" + name + "/psnr", epoch_id, epoch_loss_train_psnr);
        tblogger->add_scalar("Loss" + name + "/ssim", epoch_id, epoch_loss_train_ssim);

        if (write_gt)
        {
            std::cout << "Saving hdr volume..." << std::endl;
            SaveHDRImageTensor(volume, epoch_dir + "/volume_target_match.hdr");

            torch::Tensor volume_density = target;
            // [z, y, x]
            volume_density = volume_density.squeeze(0);
            // [3, z, y, x]
            volume_density = ColorizeTensor(volume_density, colorizeTurbo);
            // [z, 3, y, x]
            volume_density = volume_density.permute({1, 0, 2, 3});

            for (int i = 0; i < volume_density.size(0); ++i)
            {
                auto saiga_img = TensorToImage<ucvec3>(volume_density[i]);
                LogImage(tblogger.get(), saiga_img, tbname, i);
            }
        }

        std::cout << ConsoleColor::GREEN << "> Volume Loss " << name << " | SSIM " << epoch_loss_train_ssim << " PSNR "
                  << epoch_loss_train_psnr << ConsoleColor::RESET << std::endl;
    }

    double EvalStepProjection(TrainScene& ts, std::vector<int> indices, std::string name, int epoch_id,
                              std::string checkpoint_name, bool test)
    {
        auto scene           = ts.scene;
        auto tree            = ts.tree;
        auto neural_geometry = ts.neural_geometry;

        neural_geometry->train(epoch_id, false);
        scene->train(false);
        torch::NoGradGuard ngg;

        double epoch_loss_train_l1  = 0;
        double epoch_loss_train_mse = 0;
        int image_count             = 0;

        bool mult_error = false;

        int out_w = scene->cameras.front().w * params->train_params.eval_scale;
        int out_h = scene->cameras.front().h * params->train_params.eval_scale;

        int rows_per_batch = std::max(params->train_params.batch_size / out_w, 1);

        auto options = torch::data::DataLoaderOptions().batch_size(1).drop_last(false).workers(
            params->train_params.num_workers_eval);
        auto dataset     = RowRaybasedSampleDataset(indices, scene, params, out_w, out_h, rows_per_batch);
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(dataset, options);

        std::vector<torch::Tensor> projection_images(scene->frames.size());
        std::vector<torch::Tensor> target_images(scene->frames.size());
        std::vector<torch::Tensor> per_cell_loss_sum(scene->frames.size());

        for (int i : indices)
        {
            projection_images[i] = torch::zeros({1, out_h, out_w});
            target_images[i]     = torch::zeros({1, out_h, out_w});
            per_cell_loss_sum[i] = torch::zeros({(long)tree->NumNodes()}, device);
        }

        {
            // optimize network with fixed structure
            Saiga::ProgressBar bar(std::cout,
                                   name + " (" + std::to_string(out_w) + "x" + std::to_string(out_h) + ") " +
                                       std::to_string(epoch_id) + " |",
                                   out_w * out_h * indices.size(), 30, false, 1000, "ray");

            for (RowSampleData sample_data : (*data_loader))
            {
                RayList rays =
                    scene->GetRays(sample_data.pixels.uv, sample_data.pixels.image_id, sample_data.pixels.camera_id);
                SampleList all_samples =
                    tree->CreateSamplesForRays(rays, params->octree_params.max_samples_per_node, false);

                auto predicted_image =
                    neural_geometry->ComputeImage(all_samples, scene->num_channels, sample_data.NumPixels());
                predicted_image =
                    scene->tone_mapper->forward(predicted_image, sample_data.pixels.uv, sample_data.pixels.image_id);

                if (sample_data.pixels.target_mask.defined())
                {
                    // Multiply by mask so the loss of invalid pixels is 0
                    predicted_image           = predicted_image * sample_data.pixels.target_mask;
                    sample_data.pixels.target = sample_data.pixels.target * sample_data.pixels.target_mask;
                }

                CHECK_EQ(predicted_image.sizes(), sample_data.pixels.target.sizes());
                auto per_ray_loss_mse = ((predicted_image - sample_data.pixels.target)).square();

                // CHECK_EQ(image_samples.image_index.size(0), 1);
                int image_id = sample_data.image_id;

                if (projection_images[image_id].is_cpu())
                {
                    for (auto& i : projection_images)
                    {
                        if (i.defined()) i = i.cpu();
                    }
                    projection_images[image_id] = projection_images[image_id].cuda();
                    target_images[image_id]     = target_images[image_id].cuda();
                }
                auto prediction_rows = predicted_image.reshape({predicted_image.size(0), -1, out_w});
                auto target_rows = sample_data.pixels.target.reshape({sample_data.pixels.target.size(0), -1, out_w});

                projection_images[image_id].slice(1, sample_data.row_start, sample_data.row_end) = prediction_rows;
                target_images[image_id].slice(1, sample_data.row_start, sample_data.row_end)     = target_rows;

                auto [loss_sum, weight_sum] =
                    neural_geometry->AccumulateSampleLossPerNode(all_samples, per_ray_loss_mse);

                // PrintTensorInfo(loss_sum);
                // PrintTensorInfo(loss_sum2);

                per_cell_loss_sum[image_id] += loss_sum.detach();

                epoch_loss_train_mse += per_ray_loss_mse.mean().item().toFloat();

                image_count += sample_data.batch_size;

                bar.SetPostfix(" MSE=" + std::to_string(epoch_loss_train_mse / image_count));
                bar.addProgress(sample_data.NumPixels());
            }
        }

        if (!test)
        {
            torch::Tensor cell_loss_combined;
            if (mult_error)
            {
                cell_loss_combined = torch::ones({(long)tree->NumNodes()}, device);
                for (int i : indices)
                {
                    cell_loss_combined *= per_cell_loss_sum[i];
                }
            }
            else
            {
                cell_loss_combined = torch::zeros({(long)tree->NumNodes()}, device);
                for (int i : indices)
                {
                    cell_loss_combined += per_cell_loss_sum[i];
                }
            }
            tree->SetErrorForActiveNodes(cell_loss_combined, "override");
        }

        if (!projection_images.empty())
        {
            epoch_loss_train_mse = 0;
            for (int i : indices)
            {
                auto predicted_image = projection_images[i].cpu().unsqueeze(0);
                auto ground_truth    = target_images[i].cpu().unsqueeze(0);
                // auto actual_target   = img->projection;
                // PrintTensorInfo(ground_truth-actual_target);

                CHECK_EQ(ground_truth.dim(), 4);
                CHECK_EQ(predicted_image.dim(), 4);

                auto image_loss_mse = ((predicted_image - ground_truth)).square();
                auto image_loss_l1  = ((predicted_image - ground_truth)).abs();

                epoch_loss_train_mse += image_loss_mse.mean().item().toFloat();
                epoch_loss_train_l1 += image_loss_l1.mean().item().toFloat();

                auto err_col_tens = ColorizeTensor(image_loss_l1.squeeze(0).mean(0) * 4, colorizeTurbo).unsqueeze(0);
                CHECK_EQ(err_col_tens.dim(), 4);

                if (!checkpoint_name.empty())
                {
                    auto scale = 1.f / ground_truth.max();

                    ground_truth *= scale;
                    predicted_image *= scale;

                    auto predicted_colorized =
                        ColorizeTensor(predicted_image.squeeze(0).squeeze(0) * 8, colorizeTurbo).unsqueeze(0);
                    // LogImage(tblogger.get(),
                    // TensorToImage<ucvec3>(predicted_colorized),
                    //          checkpoint_name + "/Render" + name + "/render_x8", i);

                    predicted_colorized =
                        ColorizeTensor(predicted_image.squeeze(0).squeeze(0), colorizeTurbo).unsqueeze(0);
                    //                    LogImage(tblogger.get(),
                    //                    TensorToImage<ucvec3>(predicted_colorized),
                    //                             checkpoint_name + "/Render" + name +
                    //                             "/render", i);

                    auto gt_colorized = ColorizeTensor(ground_truth.squeeze(0).squeeze(0), colorizeTurbo).unsqueeze(0);

                    if (scene->num_channels == 1)
                    {
                        ground_truth    = ground_truth.repeat({1, 3, 1, 1});
                        predicted_image = predicted_image.repeat({1, 3, 1, 1});
                    }

                    auto stack    = torch::cat({gt_colorized, predicted_colorized, err_col_tens}, 0);
                    auto combined = ImageBatchToImageRow(stack);
                    if (indices.size() == 1)
                    {
                        LogImage(tblogger.get(), TensorToImage<ucvec3>(combined),
                                 checkpoint_name + "/Render" + name + "/gt_render_l1error", epoch_id);
                    }
                    else
                    {
                        LogImage(tblogger.get(), TensorToImage<ucvec3>(combined),
                                 checkpoint_name + "/Render" + name + "/gt_render_l1error", i);
                    }
                }
            }
        }
        epoch_loss_train_mse /= indices.size();
        epoch_loss_train_l1 /= indices.size();
        float epoch_loss_train_psnr = 10 * std::log10(1. / epoch_loss_train_mse);
        if (epoch_id > 0)
        {
            CHECK(std::isfinite(epoch_loss_train_psnr));
            // tblogger->add_scalar("Loss" + name + "/mse", epoch_id,
            // epoch_loss_train_mse); tblogger->add_scalar("Loss" + name + "/l1",
            // epoch_id, epoch_loss_train_l1);
            tblogger->add_scalar("Loss" + name + "/psnr", epoch_id, epoch_loss_train_psnr);
        }
        std::cout << ConsoleColor::GREEN << "> Loss " << name << " | MSE " << epoch_loss_train_mse << " L1 "
                  << epoch_loss_train_l1 << " PSNR " << epoch_loss_train_psnr << ConsoleColor::RESET << std::endl;
        return epoch_loss_train_mse;
    }

    void OptimizeTreeStructure(TrainScene& ts, int epoch_id)
    {
        {
            auto scene           = ts.scene;
            auto tree            = ts.tree;
            auto neural_geometry = ts.neural_geometry;
            torch::Tensor grid_interpolated;
            torch::Tensor old_mask;

            tree->to(torch::kCPU);

            HyperTreeStructureOptimizer opt(tree, params->octree_params.tree_optimizer_params);
            if (opt.OptimizeTree())
            {
                // If the structure has changed we need to recompute some stuff

                std::cout << "Resetting Optimizer..." << std::endl;
                neural_geometry->ResetGeometryOptimizer();
                current_lr_factor = 1;
            }
            tree->to(device);
        }
    }

    PSNR loss_function_psnr = PSNR(0, 1);

    std::shared_ptr<CombinedParams> params;
    std::string experiment_dir;

    std::vector<TrainScene> scenes;
    double current_lr_factor = 1;

   public:
    std::shared_ptr<TensorBoardLogger> tblogger;
};

template <typename ParamType>
ParamType LoadParamsHybrid(int argc, const char* argv[])
{
    CLI::App app{"Train The Hyper Tree", "hyper_train"};

    std::string config_file;
    app.add_option("config_file", config_file);
    auto params = ParamType();
    params.Load(app);
    app.parse(argc, argv);

    std::cout << "Loading Config File " << config_file << std::endl;
    params.Load(config_file);

    // params.Load(app);
    app.parse(argc, argv);

    // std::cout << app.help("", CLI::AppFormatMode::All) << std::endl;

    return params;
}

int main(int argc, const char* argv[])
{
    auto params = std::make_shared<CombinedParams>(LoadParamsHybrid<CombinedParams>(argc, argv));

    std::string experiment_dir = PROJECT_DIR.append("Experiments");
    std::filesystem::create_directories(experiment_dir);
    experiment_dir = experiment_dir + "/" + params->train_params.ExperimentString() + "/";
    std::filesystem::create_directories(experiment_dir);
    params->Save(experiment_dir + "/params.ini");

    Trainer trainer(params, experiment_dir);

    std::string args_combined;
    for (int i = 0; i < argc; ++i)
    {
        args_combined += std::string(argv[i]) + " ";
    }
    trainer.tblogger->add_text("arguments", 0, args_combined.c_str());
    trainer.Train();

    return 0;
}