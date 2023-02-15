/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the GPL v3 License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/directory.h"

#include "Settings.h"
#include "data/Dataloader.h"
#include "utils/utils.h"
#include "data/SceneBase.h"
#include "build_config.h"
#include "tensorboard_logger.h"

using namespace Saiga;



struct XTekCT : public ParamsBase
{
  SAIGA_PARAM_STRUCT(XTekCT);
  SAIGA_PARAM_STRUCT_FUNCTIONS;
  //    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override
  template <class ParamIterator>
  void Params(ParamIterator* it)
  {
        SAIGA_PARAM(VoxelsX);
        SAIGA_PARAM(VoxelsY);
        SAIGA_PARAM(VoxelsZ);
        SAIGA_PARAM(VoxelSizeX);
        SAIGA_PARAM(VoxelSizeY);
        SAIGA_PARAM(VoxelSizeZ);

        SAIGA_PARAM(DetectorPixelsX);
        SAIGA_PARAM(DetectorPixelsY);
        SAIGA_PARAM(DetectorPixelSizeX);
        SAIGA_PARAM(DetectorPixelSizeY);
        SAIGA_PARAM(SrcToObject);
        SAIGA_PARAM(SrcToDetector);
        SAIGA_PARAM(MaskRadius);
        SAIGA_PARAM(Projections);
    }

    void Print()
    {
        Table tab({30, 20});

        tab << "Name"
            << "Value";
        tab << "DetectorPixelsX" << DetectorPixelsX;
        tab << "DetectorPixelsY" << DetectorPixelsY;
        tab << "DetectorPixelSizeX" << DetectorPixelSizeX;
        tab << "DetectorPixelSizeY" << DetectorPixelSizeY;
        tab << "SrcToObject" << SrcToObject;
        tab << "SrcToDetector" << SrcToDetector;
        tab << "MaskRadius" << MaskRadius;
        tab << "Projections" << Projections;
        tab << "VoxelsX" << VoxelsX;
        tab << "VoxelsY" << VoxelsY;
        tab << "VoxelsZ" << VoxelsZ;
        tab << "VoxelSizeX" << VoxelSizeX;
        tab << "VoxelSizeY" << VoxelSizeY;
        tab << "VoxelSizeZ" << VoxelSizeZ;
    }
    // projection settings
    int DetectorPixelsX       = 0;
    int DetectorPixelsY       = 0;
    double DetectorPixelSizeX = 0;
    double DetectorPixelSizeY = 0;

    int VoxelsX = 0;
    int VoxelsY = 0;
    int VoxelsZ = 0;

    double VoxelSizeX = 0;
    double VoxelSizeY = 0;
    double VoxelSizeZ = 0;

    double SrcToObject   = 0;
    double SrcToDetector = 0;
    double MaskRadius    = 0;
    int Projections      = 0;
};



void Convert(std::string dataset_name, std::string kaust_dir, std::string out_dir, ivec2 crop_low, ivec2 crop_high, float scene_scale,
             int intensity_threshold = 0)
{
    std::filesystem::create_directories(out_dir);

    auto scene_path = std::filesystem::canonical(kaust_dir).string();
    auto scene_name = std::filesystem::path(scene_path).filename();

    std::cout << "Loading " << scene_path << std::endl;
    std::cout << "Name " << scene_name << std::endl;
    std::cout << "saving in " << out_dir << std::endl;


    DatasetParams out_params;

    Directory dir(scene_path + "/projections");
    auto image_names = dir.getFilesEnding(".tif");
    std::sort(image_names.begin(), image_names.end());


    auto ct_params = XTekCT(scene_path + "/" + dataset_name + "_CT_parameters.xtekct");



    if (crop_low(0) < 0) crop_low(0) = 0;
    if (crop_low(1) < 0) crop_low(1) = 0;
    if (crop_high(0) < 0) crop_high(0) = ct_params.DetectorPixelsX;
    if (crop_high(1) < 0) crop_high(1) = ct_params.DetectorPixelsY;

    Vec3 voxel_size  = Vec3(ct_params.VoxelSizeX, ct_params.VoxelSizeY, ct_params.VoxelSizeZ);
    ivec3 voxels     = ivec3(ct_params.VoxelsX, ct_params.VoxelsY, ct_params.VoxelsZ);
    Vec3 volume_size = voxels.cast<double>().array() * voxel_size.cast<double>().array();
    std::cout << "Volume Size " << volume_size.transpose() << std::endl;

    double max_volume_size = volume_size.array().maxCoeff();
    std::cout << "new_volume_size " << max_volume_size << std::endl;

    std::cout << "output_grid 512x512x512" << std::endl;
    std::cout << "voxel_size(512) " << max_volume_size / 512 << std::endl;

    ivec2 detector_pixels     = ivec2(ct_params.DetectorPixelsX, ct_params.DetectorPixelsY);
    Vec2 detector_pixel_size  = Vec2(ct_params.DetectorPixelSizeX, ct_params.DetectorPixelSizeY);
    Vec2 actual_detector_size = detector_pixel_size.array() * detector_pixels.cast<double>().array();
    ivec2 cropped_size        = crop_high - crop_low;

    std::cout << "Detector Pixels " << detector_pixels.transpose() << std::endl;
    std::cout << "ct_params.SrcToDetector " << ct_params.SrcToDetector << std::endl;
    std::cout << "ct_params.SrcToObject " << ct_params.SrcToObject << std::endl;
    std::cout << "Detector size " << actual_detector_size.transpose() << std::endl;

    std::cout << "Cropped Pixels  " << cropped_size.transpose() << std::endl;
    CameraBase cam;
    cam.K.cx = detector_pixels(0) / 2.f - 0.5 - crop_low(0);
    cam.K.cy = detector_pixels(1) / 2.f - 0.5 - crop_low(1);

    cam.K.fx = detector_pixels(0) / actual_detector_size(0) * (ct_params.SrcToDetector);
    cam.K.fy = detector_pixels(1) / actual_detector_size(1) * (ct_params.SrcToDetector);

    cam.w = cropped_size(0);
    cam.h = cropped_size(1);
    std::filesystem::remove(out_dir + "/camera.ini");
    cam.Save(out_dir + "/camera.ini");

    out_params.image_dir = out_dir + "/images";
    out_params.mask_dir  = out_dir + "/masks";
    std::filesystem::create_directories(out_params.image_dir);
    std::filesystem::create_directories(out_params.mask_dir);


    if (1)
    {
        std::ofstream ostream1(out_dir + "/images.txt");
        std::ofstream ostream2(out_dir + "/masks.txt");
        std::vector<std::string> new_image_names;
        std::vector<std::string> new_masks_names;
        for (int i = 0; i < image_names.size(); ++i)
        {
            std::string new_image_name = leadingZeroString(i, 4) + ".png";
            new_image_names.push_back(new_image_name);
            new_masks_names.push_back(new_image_name);
            ostream1 << new_image_name << std::endl;
            ostream2 << new_image_name << std::endl;
        }
        int min_v      = 65000;
        int max_v      = 0;
        int low_img_id = -1;

        int mask_dilation = 6;

        ProgressBar bar(std::cout, "Process Images", image_names.size(), 30, false, 10);
#pragma omp parallel for num_threads(8)
        for (int i = 0; i < image_names.size(); i++)
        //                        for (int i = 0; i < 4; ++i)
        {
            // std::cout << "Loading image " << scene_path + "/" + image_names[i] << std::endl;
            TemplatedImage<unsigned short> raw(scene_path + "/projections/" + image_names[i]);

            TemplatedImage<unsigned short> cropped;
            cropped = raw.getImageView().subImageView(crop_low(1), crop_low(0), cropped_size(1), cropped_size(0));

            TemplatedImage<unsigned char> mask(cropped.dimensions());
            for (int i : mask.rowRange())
            {
                for (int j : mask.colRange())
                {
                    mask(i, j) = (cropped(i, j) > intensity_threshold) ? 255 : 0;
                }
            }

            for (int di = 0; di < mask_dilation; ++di)
            {
                auto mask_view = mask.getImageView();
                TemplatedImage<unsigned char> new_mask(cropped.dimensions());
                for (int i : new_mask.rowRange())
                {
                    for (int j : new_mask.colRange())
                    {
                        new_mask(i, j) = mask_view(i, j) & mask_view.clampedRead(i + 1, j) &
                                         mask_view.clampedRead(i - 1, j) & mask_view.clampedRead(i, j - 1) &
                                         mask_view.clampedRead(i, j + 1);
                    }
                }
                mask = new_mask;
            }


            int img_min = 234564, img_max = 0;
            for (int i : mask.rowRange())
            {
                for (int j : mask.colRange())
                {
                    if (mask(i, j))
                    {
                        img_min = std::min<int>(img_min, cropped(i, j));
                        img_max = std::max<int>(img_max, cropped(i, j));
                    }
                }
            }
            CHECK_GT(img_min, intensity_threshold);

            cropped.save(out_params.image_dir + "/" + new_image_names[i]);
            mask.save(out_params.mask_dir + "/" + new_image_names[i]);

            if (i == 234)
            {
                auto cpy = cropped;
                for (int i : mask.rowRange())
                {
                    for (int j : mask.colRange())
                    {
                        if (mask(i, j) == 0)
                        {
                            cpy(i, j) = std::numeric_limits<unsigned short>::max();
                        }
                    }
                }
                cpy.save(out_dir + "/mask_overlay.png");
                cropped.save(out_dir + "/mask_real.png");
            }

#pragma omp critical
            {
                if (img_min < min_v)
                {
                    low_img_id = i;
                }
                min_v = std::min(min_v, img_min);
                max_v = std::max(max_v, img_max);
            }
            bar.SetPostfix("Min " + std::to_string(min_v) + " Max " + std::to_string(max_v) + " Low Id " +
                           std::to_string(low_img_id));
            bar.addProgress(1);
        }

        out_params.xray_min = min_v;
        out_params.xray_max = max_v;
    }



    // After that the volume fits in the unit cube [-1, 1]
    auto scale_factor = 2. / max_volume_size;

    {
        std::ofstream ostream2(out_dir + "/camera_indices.txt");
        std::ofstream strm2(out_dir + "/poses.txt");
        for (int i = 0; i < image_names.size(); ++i)
        {
            // the last frame has the same angle as the first one!!!
            // remove the -1 if thats not the case
            double ang = double(i) / (image_names.size() - 1) * 2 * pi<double>();

            Vec3 dir             = Vec3(sin(-ang), -cos(-ang), 0);
            Vec3 source          = ct_params.SrcToObject * scale_factor * dir;
            Vec3 detector_center = -dir * (ct_params.SrcToDetector - ct_params.SrcToObject) * scale_factor;

            Vec3 up      = Vec3(0, 0, -1);
            Vec3 forward = (detector_center - source).normalized();
            Vec3 right   = up.cross(forward).normalized();
            Mat3 R;
            R.col(0) = right;
            R.col(1) = up;
            R.col(2) = forward;

            ostream2 << 0 << std::endl;

            Quat q = Sophus::SO3d(R).unit_quaternion();
            Vec3 t = source;

            strm2 << std::scientific << std::setprecision(15);
            strm2 << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " " << t.x() << " " << t.y() << " "
                  << t.z() << "\n";
        }
    }
    out_params.scene_scale = scene_scale;
    std::filesystem::remove(out_dir + "/dataset.ini");
    out_params.Save(out_dir + "/dataset.ini");
}

int main(int argc, const char* argv[])
{

    std::string input_base  = "scenes/";
    std::string output_base = "scenes/";
    Convert("Pepper", input_base + "/Pepper", output_base + "/pepper", ivec2(50, 20), ivec2(1880, 1400), 1);
    // Convert(input_base + "/Teapot_90kV", output_base + "/teapot", ivec2(-1, -1), ivec2(-1, 1350), 1, 17500);
    return 0;
//    Convert(input_base + "/marine_decoration", output_base + "/marine_decoration", ivec2(-1, -1), ivec2(-1, 1400), 1.);
//    Convert(input_base + "/monument", output_base + "/monument", ivec2(-1, -1), ivec2(-1, 1500), 1.);
//    Convert(input_base + "/toy_car", output_base + "/toy_car", ivec2(-1, -1), ivec2(-1, 1320), 1.);
//    Convert(input_base + "/pomegranate", output_base + "/pomegranate", ivec2(-1, -1), ivec2(-1, 1400), 1.);
//
//
//
//    Convert(input_base + "/star", output_base + "/star", ivec2(-1, -1), ivec2(-1, 1350), 1., 25000);
//    Convert(input_base + "/Plastic_flower", output_base + "/Plastic_flower", ivec2(-1, -1), ivec2(-1, 1300), 1, 20000);
//
//    Convert(input_base + "/Chest", output_base + "/Chest", ivec2(-1, -1), ivec2(-1, -1), 1.);
//    Convert(input_base + "/Fan", output_base + "/Fan", ivec2(-1, -1), ivec2(-1, -1), 1.);
//    Convert(input_base + "/Fruit", output_base + "/Fruit", ivec2(-1, -1), ivec2(-1, 1470), 1.);
//    Convert(input_base + "/RopeBall", output_base + "/RopeBall", ivec2(0, 0), ivec2(-1, 1450), 1.);
//    Convert(input_base + "/Textile_flower", output_base + "/Textile_flower", ivec2(-1, -1), ivec2(-1, 1350), 1.);

    return 0;
}
