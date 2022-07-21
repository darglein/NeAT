/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the GPL v3 License.
 * See LICENSE file for more information.
 */
#include "SceneBase.h"

#include "saiga/core/util/ProgressBar.h"
#include "saiga/vision/torch/ColorizeTensor.h"

#include "utils/cimg_wrapper.h"
vec2 PixelToUV(int x, int y, int w, int h)
{
    CHECK_GE(x, 0);
    CHECK_GE(y, 0);
    ivec2 shape(h, w);
    CHECK_LT(x, shape(1));
    CHECK_LT(y, shape(0));
    vec2 s  = vec2(shape(1) - 1, shape(0) - 1);
    vec2 uv = (vec2(x, y)).array() / s.array();
    return uv;
}

vec2 UVToPix(vec2 uv, int w, int h)
{
    ivec2 shape(h, w);
    vec2 s   = vec2(shape(1) - 1, shape(0) - 1);
    vec2 pix = (uv).array() * s.array();
    return pix;
}

torch::Tensor UnifiedImage::CoordinatesRandomNoInterpolate(int count, int w, int h)
{
    auto y = torch::randint(0, h, {count, 1}).to(torch::kInt).to(torch::kFloat);
    auto x = torch::randint(0, w, {count, 1}).to(torch::kInt).to(torch::kFloat);

    torch::Tensor coords = torch::cat({x, y}, 1);
    return PixelToUV(coords, w, h, uv_align_corners);
}
torch::Tensor UnifiedImage::CoordinatesRow(int row_start, int row_end, int w, int h)
{
    int num_rows = row_end - row_start;
    CHECK_GT(num_rows, 0);
    CHECK_LE(row_end, h);

    torch::Tensor px_coords = torch::empty({w * num_rows, 2});
    vec2* pxs               = px_coords.data_ptr<vec2>();

    for (int row_id = row_start; row_id < row_end; ++row_id)
    {
        for (int x = 0; x < w; ++x)
        {
            pxs[(row_id - row_start) * w + x] = vec2(x, row_id);
        }
    }

    return PixelToUV(px_coords, w, h, uv_align_corners);
}
std::pair<torch::Tensor, torch::Tensor> UnifiedImage::SampleProjection(torch::Tensor uv)
{
    torch::nn::functional::GridSampleFuncOptions opt;
    opt.align_corners(uv_align_corners).padding_mode(torch::kBorder).mode(torch::kBilinear);
    uv = uv * 2 - 1;
    uv = uv.unsqueeze(0).unsqueeze(0);
    auto samples = torch::nn::functional::grid_sample(projection.unsqueeze(0), uv, opt);
    samples      = samples.reshape({NumChannels(), -1});

    torch::Tensor samples_mask;
    if (mask.defined())
    {
        samples_mask = torch::nn::functional::grid_sample(mask.unsqueeze(0), uv, opt);
        samples_mask = samples_mask.reshape({1, -1});
    }

    return {samples, samples_mask};
}

void SceneBase::save(std::string dir) {}
void SceneBase::Finalize()
{
    std::vector<SE3> poses;
    for (auto& f : frames)
    {
        poses.push_back(f->pose);
    }
    std::vector<IntrinsicsPinholed> intrinsics;
    for (auto c : cameras)
    {
        intrinsics.push_back(c.K);
    }
    pose         = CameraPoseModule(poses);
    camera_model = CameraModelModule(cameras.front().h, cameras.front().w, intrinsics);

    pose->to(device);
    camera_model->to(device);

    CHECK(params);
    CHECK(camera_model);
    tone_mapper = PhotometricCalibration(frames.size(), cameras.size(), camera_model->h, camera_model->w,
                                         params->photo_calib_params);
    tone_mapper->to(device);

    {
        std::vector<torch::optim::OptimizerParamGroup> g;
        if (params->train_params.optimize_pose)
        {
            g.emplace_back(pose->parameters(),
                           std::make_unique<torch::optim::SGDOptions>(params->train_params.lr_pose));
        }
        if (params->train_params.optimize_intrinsics)
        {
            g.emplace_back(camera_model->parameters(),
                           std::make_unique<torch::optim::SGDOptions>(params->train_params.lr_intrinsics));
        }

        if (!g.empty())
        {
            structure_optimizer = std::make_shared<torch::optim::SGD>(g, torch::optim::SGDOptions(10));
        }
    }
    {
        std::vector<torch::optim::OptimizerParamGroup> g;

        if (params->photo_calib_params.exposure_enable)
        {
            std::cout << "Optimizing Tone Mapper Exposure LR " << params->photo_calib_params.exposure_lr << std::endl;
            std::vector<torch::Tensor> t;
            t.push_back(tone_mapper->exposure_bias);
            g.emplace_back(t, std::make_unique<torch::optim::SGDOptions>(params->photo_calib_params.exposure_lr));

            if (params->photo_calib_params.exposure_mult)
            {
                std::cout << "Optimizing Tone Mapper Exposure Factor LR " << params->photo_calib_params.exposure_lr
                          << std::endl;
                t.clear();
                t.push_back(tone_mapper->exposure_factor);
                g.emplace_back(t, std::make_unique<torch::optim::SGDOptions>(params->photo_calib_params.exposure_lr));
            }
        }
        if (params->photo_calib_params.sensor_bias_enable)
        {
            std::cout << "Optimizing Tone Mapper Sensor Bias LR " << params->photo_calib_params.sensor_bias_lr
                      << std::endl;
            std::vector<torch::Tensor> t;
            t.push_back(tone_mapper->sensor_bias);
            g.emplace_back(t, std::make_unique<torch::optim::SGDOptions>(params->photo_calib_params.sensor_bias_lr));
        }
        if (tone_mapper->response)
        {
            std::cout << "Optimizing Sensor Response Bias LR " << params->photo_calib_params.response_lr << std::endl;
            std::vector<torch::Tensor> t = tone_mapper->response->parameters();
            g.emplace_back(t, std::make_unique<torch::optim::SGDOptions>(params->photo_calib_params.response_lr));
        }
        if (!g.empty())
        {
            tm_optimizer = std::make_shared<torch::optim::SGD>(g, torch::optim::SGDOptions(10));
        }
    }

    std::sort(train_indices.begin(), train_indices.end());
    std::sort(test_indices.begin(), test_indices.end());
    active_train_images =
        torch::from_blob(train_indices.data(), {(long)train_indices.size()}, torch::TensorOptions(torch::kInt32))
            .clone()
            .cuda();
    active_test_images =
        torch::from_blob(test_indices.data(), {(long)test_indices.size()}, torch::TensorOptions(torch::kInt32))
            .clone()
            .cuda();
}
RayList SceneBase::GetRays(torch::Tensor uv, torch::Tensor image_id, torch::Tensor camera_id)
{
    RayList result;
    result.Allocate(uv.size(0), 3);
    result.to(uv.device());


    auto px_coords = UVToPixel(uv, camera_model->w, camera_model->h, uv_align_corners);
    auto unproj =
        camera_model->Unproject(camera_id, px_coords, torch::ones({uv.size(0)}, torch::TensorOptions(uv.device())));
    unproj = unproj / torch::norm(unproj, 2, 1, true);

    auto dir2 = pose->RotatePoint(unproj, image_id);
    CHECK_EQ(dir2.requires_grad(), unproj.requires_grad());

    result.origin    = torch::index_select(pose->translation, 0, image_id).to(torch::kFloat32);
    result.direction = dir2;

    result.origin    = result.origin.to(torch::kFloat32);
    result.direction = result.direction.to(torch::kFloat32);

    return result;
}

void SceneBase::PrintGradInfo(int epoch_id, TensorBoardLogger* logger)
{
    {
        auto t = pose->rotation_tangent;
        std::vector<double> mean;

        if (t.grad().defined())
        {
            mean.push_back(t.grad().abs().mean().item().toFloat());
        }

        logger->add_scalar("Gradient/" + scene_name + "/rotation", epoch_id, Statistics(mean).mean);
    }
    {
        auto t = pose->translation;
        std::vector<double> mean;

        if (t.grad().defined())
        {
            mean.push_back(t.grad().abs().mean().item().toFloat());
        }

        logger->add_scalar("Gradient/" + scene_name + "/translation", epoch_id, Statistics(mean).mean);
    }
    {
        auto params = camera_model->parameters();
        std::vector<double> mean;
        for (auto t : params)
        {
            if (t.grad().defined())
            {
                mean.push_back(t.grad().abs().mean().item().toFloat());
            }
        }
        logger->add_scalar("Gradient/" + scene_name + "/intrinsics", epoch_id, Statistics(mean).mean);
    }
}
void SceneBase::OptimizerStep(int epoch_id, bool only_image_params)
{
    if (structure_optimizer)
    {
        if (epoch_id > params->train_params.optimize_structure_after_epochs)
        {
            structure_optimizer->step();
            pose->ApplyTangent();
        }
        structure_optimizer->zero_grad();
    }
    if (tm_optimizer)
    {
        if (only_image_params)
        {
            if (tone_mapper->params.sensor_bias_enable)
            {
                tone_mapper->sensor_bias.mutable_grad().zero_();
            }
            if (tone_mapper->response)
            {
                tone_mapper->response->response.mutable_grad().zero_();
            }
        }

        if (epoch_id > params->train_params.optimize_tone_mapper_after_epochs)
        {
            tm_optimizer->step();
        }
        tm_optimizer->zero_grad();
        tone_mapper->ApplyConstraints();
    }
}
void SceneBase::PrintInfo(int epoch_id, TensorBoardLogger* logger)
{
    std::cout << std::left;
    if (tone_mapper->params.exposure_enable && tone_mapper->params.exposure_lr > 0)
    {
        auto selected_bias_train = torch::index_select(tone_mapper->exposure_bias, 0, active_train_images).cpu();
        auto selected_bias_test  = torch::index_select(tone_mapper->exposure_bias, 0, active_test_images).cpu();
        for (int i = 0; i < selected_bias_train.size(0); ++i)
        {
            logger->add_scalar("ToneMapper/" + scene_name + "/exp_bias_train", i,
                               selected_bias_train.data_ptr<float>()[i]);
        }
        for (int i = 0; i < selected_bias_test.size(0); ++i)
        {
            logger->add_scalar("ToneMapper/" + scene_name + "/exp_bias_test", i,
                               selected_bias_test.data_ptr<float>()[i]);
        }
    }
    if (tone_mapper->params.sensor_bias_enable && tone_mapper->params.sensor_bias_lr > 0)
    {
        auto bias_image = tone_mapper->sensor_bias;
        // std::cout << std::setw(30) << "Sensor Bias:" << TensorInfo(bias_image) << std::endl;

        auto err_col_tens = ColorizeTensor(bias_image.squeeze(0).squeeze(0) * 8, colorizeTurbo);
        LogImage(logger, TensorToImage<ucvec3>(err_col_tens), "Tonemapper/" + scene_name + "/sensor_bias_x8", epoch_id);
    }


    if (tone_mapper->response)
    {
        auto crfs = tone_mapper->response->GetCRF().front();
        LogImage(logger, crfs.Image(), "Tonemapper/" + scene_name + "/sensor_response", epoch_id);
    }
}
SceneBase::SceneBase(std::string _scene_dir)
{
    scene_path = std::filesystem::canonical(_scene_dir).string();
    scene_name = std::filesystem::path(scene_path).filename();

    std::cout << "====================================" << std::endl;
    std::cout << "Scene Base" << std::endl;
    std::cout << "  Name         " << scene_name << std::endl;
    std::cout << "  Path         " << scene_path << std::endl;
    SAIGA_ASSERT(!scene_name.empty());
    CHECK(std::filesystem::exists(scene_path));
    CHECK(std::filesystem::exists(scene_path + "/dataset.ini"));

    auto file_pose           = scene_path + "/poses.txt";
    auto file_image_names    = scene_path + "/images.txt";
    auto file_mask_names     = scene_path + "/masks.txt";
    auto file_camera_indices = scene_path + "/camera_indices.txt";

    dataset_params = DatasetParams(scene_path + "/dataset.ini");

    {
        CameraBase cam(scene_path + "/camera.ini");
        CHECK_GT(cam.w, 0);
        CHECK_GT(cam.h, 0);
        cameras.push_back(cam);
    }


    if (!dataset_params.volume_file.empty())
    {
        CHECK(std::filesystem::exists(scene_path + "/" + dataset_params.volume_file));
        torch::load(ground_truth_volume, scene_path + "/" + dataset_params.volume_file);
        std::cout << "Ground Truth Volume " << TensorInfo(ground_truth_volume) << std::endl;
    }

    std::vector<Sophus::SE3d> poses;
    if (std::filesystem::exists(file_pose))
    {
        std::ifstream strm(file_pose);

        std::string line;
        while (std::getline(strm, line))
        {
            std::stringstream sstream(line);
            Quat q;
            Vec3 t;
            sstream >> q.x() >> q.y() >> q.z() >> q.w() >> t.x() >> t.y() >> t.z();
            poses.push_back({q, t});
        }
    }


    std::vector<std::string> images;
    std::vector<std::string> masks;


    if (std::filesystem::exists(file_image_names))
    {
        std::ifstream strm(file_image_names);

        std::string line;
        while (std::getline(strm, line))
        {
            images.push_back(line);
        }
    }

    if (std::filesystem::exists(file_mask_names))
    {
        std::ifstream strm(file_mask_names);

        std::string line;
        while (std::getline(strm, line))
        {
            masks.push_back(line);
        }
    }

    std::vector<int> camera_indices;
    if (std::filesystem::exists(file_camera_indices))
    {
        std::ifstream strm(file_camera_indices);

        std::string line;
        while (std::getline(strm, line))
        {
            camera_indices.push_back(to_int(line));
        }
    }


    int n_frames = std::max({images.size(), poses.size()});
    frames.resize(n_frames);


    SAIGA_ASSERT(!poses.empty());
    SAIGA_ASSERT(masks.empty() || masks.size() == frames.size());
    SAIGA_ASSERT(camera_indices.empty() || camera_indices.size() == frames.size());

    for (int i = 0; i < n_frames; ++i)
    {
        int camera_id = camera_indices.empty() ? 0 : camera_indices[i];
        auto img      = std::make_shared<UnifiedImage>();

        img->camera_id          = camera_id;
        img->pose               = poses[i];
        img->pose.translation() = img->pose.translation() * dataset_params.scene_scale;
        img->image_file         = images[i];
        if (!masks.empty()) img->mask_file = masks[i];

        frames[i] = img;
    }


    std::vector<double> distances;
    for (auto& img : frames)
    {
        auto d = img->pose.translation().norm();
        distances.push_back(d);
    }
    std::cout << "  Avg distance " << Statistics(distances).mean << std::endl;

    std::cout << "  Volume Scale " << dataset_params.scene_scale << std::endl;
    std::cout << "  Images       " << frames.size() << std::endl;
    std::cout << "  Img. Size    " << cameras.front().w << " x " << cameras.front().h << std::endl;
    std::cout << "====================================" << std::endl;
}

void SceneBase::LoadImagesCT(std::vector<int> indices)
{
    std::cout << "log input " << dataset_params.log_space_input
              << " Log10 conversion: " << dataset_params.use_log10_conversion << std::endl;
    ProgressBar bar(std::cout, "Load images", indices.size());
    for (int i = 0; i < indices.size(); ++i)
    {
        int image_id = indices[i];
        auto& img    = frames[image_id];
        if (img->projection.defined()) continue;
        TemplatedImage<unsigned short> raw(dataset_params.image_dir + "/" + img->image_file);
        TemplatedImage<float> converted(raw.dimensions());
        for (auto i : raw.rowRange())
        {
            for (auto j : raw.colRange())
            {
                converted(i, j) = raw(i, j);
            }
        }

        CHECK_EQ(converted.h, cameras[0].h);
        CHECK_EQ(converted.w, cameras[0].w);

        auto raw_tensor = ImageViewToTensor(converted.getImageView());

        if (dataset_params.log_space_input)
        {
            img->projection = raw_tensor / dataset_params.xray_max;
        }
        else
        {
            // log10 / loge conversion
            if (dataset_params.use_log10_conversion)
            {
                // convert transmittance to absorption
                img->projection = -torch::log10(raw_tensor / dataset_params.xray_max);

                img->projection = img->projection * (dataset_params.projection_factor /
                                                     -std::log10(dataset_params.xray_min / dataset_params.xray_max));
            }
            else
            {
                // convert transmittance to absorption
                img->projection = -torch::log(raw_tensor / dataset_params.xray_max);

                img->projection = img->projection * (dataset_params.projection_factor /
                                                     -std::log(dataset_params.xray_min / dataset_params.xray_max));
            }
        }

        if (!img->mask_file.empty() && !dataset_params.mask_dir.empty())
        {
            auto mask_file = dataset_params.mask_dir + "/" + img->mask_file;
            CHECK(std::filesystem::exists(mask_file)) << mask_file;

            TemplatedImage<unsigned char> mask(mask_file);
            img->mask = ImageViewToTensor(mask.getImageView());
        }

        bar.addProgress(1);
    }
}



void SceneBase::Draw(TensorBoardLogger* logger)
{
    std::string log_name = "Input/" + scene_name + "/";

    double max_radius = 0;
    for (auto& img_ : frames)
    {
        auto img   = (img_);
        max_radius = std::max(img->pose.translation().norm(), max_radius);
    }

    double scale = 1.0 / max_radius * 0.9;


    std::cout << "Drawing cone scene 2d scale = " << scale << std::endl;
    TemplatedImage<ucvec3> target_img(512, 512);
    target_img.makeZero();

    auto normalized_to_target = [&](Vec2 p) -> Vec2
    {
        p = p * scale;
        p = ((p + Vec2::Ones()) * 0.5).array() * Vec2(target_img.h - 1, target_img.w - 1).array();

        Vec2 px = p.array().round().cast<double>();

        std::swap(px(0), px(1));
        return px;
    };


    auto draw_normalized_line = [&](auto& img, Vec2 p1, Vec2 p2, ucvec3 color)
    {
        Vec2 p11 = normalized_to_target(p1);
        Vec2 p22 = normalized_to_target(p2);

        std::swap(p11(0), p11(1));
        std::swap(p22(0), p22(1));

        ImageDraw::drawLineBresenham(img.getImageView(), p11.cast<float>(), p22.cast<float>(), color);
    };

    auto draw_normalized_circle = [&](auto& img, Vec2 p1, double r, ucvec3 color)
    {
        auto p = normalized_to_target(p1);
        ImageDraw::drawCircle(img.getImageView(), vec2(p(1), p(0)), r, color);
    };



    auto draw_axis = [&](auto& img)
    {
        int rad = 2;
        for (int r = -rad; r <= rad; ++r)
        {
            Vec2 p1 = normalized_to_target(Vec2(0, -1)) + Vec2(r, 0);
            Vec2 p2 = normalized_to_target(Vec2(0, 1)) + Vec2(r, 0);

            Vec2 p3 = normalized_to_target(Vec2(-1, 0)) + Vec2(0, r);
            Vec2 p4 = normalized_to_target(Vec2(1, 0)) + Vec2(0, r);

            ImageDraw::drawLineBresenham(img.getImageView(), p1.cast<float>(), p2.cast<float>(), ucvec3(0, 255, 0));
            ImageDraw::drawLineBresenham(img.getImageView(), p3.cast<float>(), p4.cast<float>(), ucvec3(255, 0, 0));
        }

        draw_normalized_line(img, Vec2(-1, -1), Vec2(1, -1), ucvec3(255, 255, 255));
        draw_normalized_line(img, Vec2(-1, -1), Vec2(-1, 1), ucvec3(255, 255, 255));
        draw_normalized_line(img, Vec2(1, 1), Vec2(1, -1), ucvec3(255, 255, 255));
        draw_normalized_line(img, Vec2(1, 1), Vec2(-1, 1), ucvec3(255, 255, 255));
    };


    int log_count = 0;
    {
        // draw some debug stuff
        auto img_cpy = target_img;
        draw_axis(img_cpy);

        for (int i = 0; i < frames.size(); ++i)
        {
            auto img     = frames[i];
            ucvec3 color = ucvec3(255, 255, 255);

            if (std::find(train_indices.begin(), train_indices.end(), i) != train_indices.end())
            {
                color = ucvec3(0, 255, 0);
            }
            else if (std::find(test_indices.begin(), test_indices.end(), i) != test_indices.end())
            {
                color = ucvec3(255, 0, 0);
            }
            else
            {
                continue;
            }


            draw_normalized_circle(img_cpy, img->pose.translation().head<2>(), 5, color);
        }
        // img_cpy.save(output_dir + "geom_all_image_planes.png");
        LogImage(logger, img_cpy, log_name + "overview", log_count++);
    }

    int drawn_images = 0;
    for (int i = 0; i < frames.size() && drawn_images < 5; ++i)
    {
        auto img = frames[i];
        if (!img->projection.defined()) continue;
        if (logger)
        {
            auto proj = TensorToImage<unsigned char>(img->projection);
            LogImage(logger, proj, log_name + "processed", i);



            auto err_col_tens = ColorizeTensor((img->projection - img->projection.min()).squeeze(0) * 8, colorizeTurbo);
            auto proj_ampli   = TensorToImage<ucvec3>(err_col_tens);
            LogImage(logger, proj_ampli, log_name + "amp_x8_minus_min", i);

            if (img->mask.defined())
            {
                auto mask = TensorToImage<unsigned char>(img->mask);
                LogImage(logger, mask, log_name + "mask", i);
            }
            drawn_images++;
        }
    }
}
void SceneBase::InitializeBiasWithBackground(TensorBoardLogger* logger)
{
    torch::NoGradGuard ngg;
    CHECK(tone_mapper);
    auto new_exp_bias = torch::ones_like(tone_mapper->exposure_bias).cpu() * -1;

    double avg    = 0;
    int avg_count = 0;
    for (int i = 0; i < frames.size(); ++i)
    {
        auto img = frames[i];
        if (!img->projection.defined()) continue;

        // We just take the median of the top left corner.
        auto crop                         = img->projection.slice(1, 0, 64).slice(2, 0, 64);
        float median                      = crop.median().item().toFloat();
        new_exp_bias.data_ptr<float>()[i] = median;
        logger->add_scalar("ToneMapper/" + scene_name + "/exp_bias_init", i, median);
        avg += median;
        avg_count++;
    }



    // set all other frames to the avg bias
    for (int i = 0; i < frames.size(); ++i)
    {
        if (new_exp_bias.data_ptr<float>()[i] < 0)
        {
            new_exp_bias.data_ptr<float>()[i] = avg / avg_count;
        }
    }
    tone_mapper->exposure_bias.set_data(new_exp_bias.to(tone_mapper->exposure_bias.device()));
}
torch::Tensor SceneBase::SampleGroundTruth(torch::Tensor global_coordinates)
{
    CHECK_EQ(global_coordinates.dim(), 2);
    CHECK(ground_truth_volume.defined());
    torch::nn::functional::GridSampleFuncOptions opt;
    opt.align_corners(true).padding_mode(torch::kBorder).mode(torch::kBilinear);

    global_coordinates = global_coordinates.unsqueeze(0).unsqueeze(0).unsqueeze(0);

    //    PrintTensorInfo(global_coordinates);
    //    PrintTensorInfo(ground_truth_volume);

    // [batches, num_features, 1, 1, batch_size]
    auto samples = torch::nn::functional::grid_sample(ground_truth_volume.unsqueeze(0), global_coordinates, opt);

    samples = samples.squeeze(0).squeeze(1).squeeze(1);

    // samples = samples.permute({1, 0});


    return samples;
}
void SceneBase::SaveCheckpoint(const std::string& dir)
{
    auto prefix = dir + "/" + scene_name + "_";
    if (pose)
    {
        torch::save(pose, prefix + "pose.pth");
    }
    if (camera_model)
    {
        torch::save(camera_model, prefix + "camera_model.pth");
    }

    if (tone_mapper)
    {
        torch::save(tone_mapper, prefix + "tone_mapper.pth");
    }
}
