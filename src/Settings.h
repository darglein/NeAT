/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the GPL v3 License.
 * See LICENSE file for more information.
 */
#pragma once

#include "saiga/core/image/freeimage.h"
#include "saiga/core/time/time.h"
#include "saiga/core/util/ProgressBar.h"
#include "saiga/vision/torch/ImageTensor.h"
#include "saiga/vision/torch/TrainParameters.h"

#include "modules/ToneMapper.h"
#include "structure/HyperTreeStructureOptimizer.h"

#include "tensorboard_logger.h"
using namespace Saiga;



#include "build_config.h"

inline torch::Device device = torch::kCUDA;

struct Netparams : public ParamsBase
{
  SAIGA_PARAM_STRUCT(Netparams);
  SAIGA_PARAM_STRUCT_FUNCTIONS;
  //    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override
  template <class ParamIterator>
  void Params(ParamIterator* it)
  {
        SAIGA_PARAM(grid_size);
        SAIGA_PARAM(grid_features);
        SAIGA_PARAM(last_activation_function);
        SAIGA_PARAM(softplus_beta);

        SAIGA_PARAM(decoder_skip);
        SAIGA_PARAM(decoder_lr);
        SAIGA_PARAM(decoder_activation);
        SAIGA_PARAM(decoder_hidden_layers);
        SAIGA_PARAM(decoder_hidden_features);
    }

    int grid_size              = 17;
    int grid_features          = 4;

    // relu, id, abs
    std::string last_activation_function = "softplus";
    float softplus_beta                  = 2;

    bool decoder_skip                = false;
    float decoder_lr                 = 0.00005;
    std::string decoder_activation   = "silu";
    int decoder_hidden_layers        = 1;
    int decoder_hidden_features      = 64;
};

// Params for the HyperTree
struct OctreeParams : public ParamsBase
{
  SAIGA_PARAM_STRUCT(OctreeParams);
  SAIGA_PARAM_STRUCT_FUNCTIONS;
  //    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override
  template <class ParamIterator>
  void Params(ParamIterator* it)
  {
        SAIGA_PARAM(start_layer);
        SAIGA_PARAM(tree_depth);
        SAIGA_PARAM(optimize_structure);
        SAIGA_PARAM(max_samples_per_node);
        SAIGA_PARAM(culling_start_epoch);
        SAIGA_PARAM(node_culling);
        SAIGA_PARAM(culling_threshold);


        SAIGA_PARAM(tree_optimizer_params.use_saved_errors);
        SAIGA_PARAM(tree_optimizer_params.max_active_nodes);
        SAIGA_PARAM(tree_optimizer_params.error_merge_factor);
        SAIGA_PARAM(tree_optimizer_params.error_split_factor);
        SAIGA_PARAM(tree_optimizer_params.verbose);
    }

    int start_layer         = 3;
    int tree_depth          = 4;
    bool optimize_structure = true;

    int max_samples_per_node = 32;

    int culling_start_epoch = 4;
    bool node_culling       = true;

    // 0.01 for mean, 0.4 for max
    float culling_threshold = 0.1;


    TreeOptimizerParams tree_optimizer_params;
};

struct DatasetParams : public ParamsBase
{
  SAIGA_PARAM_STRUCT(DatasetParams);
  SAIGA_PARAM_STRUCT_FUNCTIONS;
  //    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override
  template <class ParamIterator>
  void Params(ParamIterator* it)
  {
        SAIGA_PARAM(image_dir);
        SAIGA_PARAM(mask_dir);
        SAIGA_PARAM(projection_factor);
        SAIGA_PARAM(vis_volume_intensity_factor);
        SAIGA_PARAM(scene_scale);
        SAIGA_PARAM(xray_min);
        SAIGA_PARAM(xray_max);
        SAIGA_PARAM(volume_file);
        SAIGA_PARAM(log_space_input);
        SAIGA_PARAM(use_log10_conversion);
    }

    // only set if a ground truth volume exists
    std::string volume_file = "";


    // linear multiplier to the projection
    // otherwise it is just transformed by xray/min/max parameters
    double projection_factor = 1;


    // Only for visualization!
    // multiplied to the intensity of the projection (after normalization)
    double vis_volume_intensity_factor = 1;

    // the camera position is multiplied by this factor to "scale" the scene
    double scene_scale = 1;

    std::string image_dir = "";
    std::string mask_dir  = "";

    // "real" raw xray is usually NOT in log space (background is white)
    // if the data is already preprocessed and converted to log space (background is black)
    // set this flag in the dataset
    bool log_space_input = false;

    // pepper:13046, 65535
    // ropeball: 26000, 63600
    double xray_min = 0;
    double xray_max = 65535;


    // true: log10
    // false: loge
    bool use_log10_conversion = true;
};

struct MyTrainParams : public TrainParams
{
  using ParamStructType = MyTrainParams;
  //    MyTrainParams() {}
  //    MyTrainParams(const std::string file) { Load(file); }

  //    using ParamStructType = MyTrainParams;
  // SAIGA_PARAM_STRUCT(MyTrainParams);

  MyTrainParams(){}
  MyTrainParams(const std::string file) : TrainParams(file) {}


  SAIGA_PARAM_STRUCT_FUNCTIONS;

  //    SAIGA_PARAM_STRUCT_FUNCTIONS;

  //    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override
  template <class ParamIterator>
  void Params(ParamIterator* it)
  {
        TrainParams::Params(it);

        SAIGA_PARAM(scene_dir);
        SAIGA_PARAM_LIST2(scene_name, ' ');
        SAIGA_PARAM(split_name);

        SAIGA_PARAM(optimize_structure_every_epochs);
        SAIGA_PARAM(optimize_structure_convergence);
        SAIGA_PARAM(per_node_batch_size);
        SAIGA_PARAM(rays_per_image);
        SAIGA_PARAM(output_volume_size);

        SAIGA_PARAM(lr_exex_grid_adam);
        SAIGA_PARAM(optimize_pose);
        SAIGA_PARAM(optimize_intrinsics);
        SAIGA_PARAM(lr_pose);
        SAIGA_PARAM(lr_intrinsics);
        SAIGA_PARAM(lr_decay_factor);
        SAIGA_PARAM(optimize_structure_after_epochs);
        SAIGA_PARAM(optimize_tree_structure_after_epochs);
        SAIGA_PARAM(optimize_tone_mapper_after_epochs);
        SAIGA_PARAM(init_bias_with_bg);
        SAIGA_PARAM(grid_init);
        SAIGA_PARAM(loss_tv);
        SAIGA_PARAM(loss_edge);
        SAIGA_PARAM(eval_scale);

    }

    std::string scene_dir               = "";
    std::vector<std::string> scene_name = {"pepper"};
    std::string split_name              = "exp_uniform_50";

    int optimize_structure_every_epochs  = 1;
    float optimize_structure_convergence = 0.95;

    std::string grid_init = "uniform";

    int rays_per_image                          = 500000;
    int per_node_batch_size                     = 256;
    int output_volume_size                      = 256;

    double lr_decay_factor = 0.95;

    double eval_scale = 1;

    double loss_tv                = 1e-4;
    double loss_edge              = 1e-3;


    float lr_exex_grid_adam = 0.04;

    // On each image we compute the median value of a top right corner crop
    // This is used to initialize the tone-mapper's bias value
    bool init_bias_with_bg = true;

    // In the first few epochs we keep the camera pose/model fixed!
    int optimize_tree_structure_after_epochs = 1;
    int optimize_structure_after_epochs      = 3;
    int optimize_tone_mapper_after_epochs    = 1;
    bool optimize_pose                       = true;
    bool optimize_intrinsics                 = true;
    float lr_pose                            = 0.001;
    float lr_intrinsics                      = 100;
};



struct CombinedParams
{
    MyTrainParams train_params;
    OctreeParams octree_params;
    Netparams net_params;
    PhotometricCalibrationParams photo_calib_params;

    CombinedParams() {}
    CombinedParams(const std::string& combined_file)
        : train_params(combined_file),
          octree_params(combined_file),
          net_params(combined_file),
          photo_calib_params(combined_file)
    {
    }

    void Save(const std::string file)
    {
        train_params.Save(file);
        octree_params.Save(file);
        net_params.Save(file);
        photo_calib_params.Save(file);
    }

    void Load(std::string file)
    {
        train_params.Load(file);
        octree_params.Load(file);
        net_params.Load(file);
        photo_calib_params.Load(file);
    }

    void Load(CLI::App& app)
    {
        train_params.Load(app);
        octree_params.Load(app);
        net_params.Load(app);
        photo_calib_params.Load(app);
    }
};
