import numpy as np
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt


class EvaluationMetrics(object):
    """ Class for evaluation metrics for segmentation"""
    def __init__(self, ground_truth: sitk.Image, segmentation: [sitk.Image]):
        self.segmented_images: [sitk.Image] = segmentation
        self.ground_truth_image: sitk.Image = ground_truth
        self.verify_same_space_image()
        self.overlap_measures = {'jacquard': 0, 'dice': 1,
                                 'volume_similarity': 2, 'false_negative': 3, 'false_positive': 4}
        self.surface_distance_measures = {'hausdorff_distance': 0, 'mean_surface_distance': 1,
                                          'median_surface_distance': 2, 'std_surface_distance': 3,
                                          'max_surface_distance': 4}
        self.overlap_results = np.zeros((len(self.segmented_images), len(self.overlap_measures)))
        self.surface_distance_results = np.zeros((len(self.segmented_images), len(self.surface_distance_measures)))

    def verify_same_space_image(self):
        for count, seg in enumerate(self.segmented_images):
            if(seg.GetOrigin() != self.ground_truth_image.GetOrigin()):
                seg.CopyInformation(self.ground_truth_image)
                self.segmented_images[count] = seg

    def set_segmented_images(self, images: [sitk.Image]):
        self.segmented_images = images

    def set_ground_truth_image(self, image: sitk.Image):
        self.ground_truth_image = image

    def flush(self):
        del self.segmented_images
        self.segmented_images = None
        del self.ground_truth_image
        self.ground_truth_image = None

    # Empty numpy arrays to hold the results
    def calculate_statistics(self):
        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        # Use the absolute values of the distance map to compute the surface distances (distance map sign, outside or
        # inside relationship, is irrelevant)
        # label = 1
        reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(self.ground_truth_image, squaredDistance=False,
                                                                       useImageSpacing=True))
        reference_surface = sitk.LabelContour(self.ground_truth_image)
        for i, segment in enumerate(self.segmented_images):
            # Overlap measures
            if np.argwhere(sitk.GetArrayFromImage(segment)).any():

                overlap_measures_filter.Execute(self.ground_truth_image, segment)
                self.overlap_results[i, self.overlap_measures['jacquard']] = overlap_measures_filter.\
                    GetJaccardCoefficient()
                self.overlap_results[i, self.overlap_measures['dice']] = overlap_measures_filter.GetDiceCoefficient()
                self.overlap_results[i, self.overlap_measures['volume_similarity']] = \
                    overlap_measures_filter.GetVolumeSimilarity()
                self.overlap_results[i, self.overlap_measures['false_negative']] = \
                    overlap_measures_filter.GetFalseNegativeError()
                self.overlap_results[i, self.overlap_measures['false_positive']] =\
                    overlap_measures_filter.GetFalsePositiveError()

                # Hausdorff distance
                hausdorff_distance_filter.Execute(self.ground_truth_image, segment)
                statistics_image_filter = sitk.StatisticsImageFilter()
                # Get the number of pixels in the reference surface by counting all pixels that are 1.
                statistics_image_filter.Execute(self.ground_truth_image)
                num_reference_surface_pixels = int(statistics_image_filter.GetSum())
                self.surface_distance_results[i, self.surface_distance_measures['hausdorff_distance']]\
                    = hausdorff_distance_filter.GetHausdorffDistance()
                # Symmetric surface distance measures
                segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(segment,
                                                                               squaredDistance=False, useImageSpacing=True))
                segmented_surface = sitk.LabelContour(segment)

                # Multiply the binary surface segmentations with the distance maps. The resulting distance
                # maps contain non-zero values only on the surface (they can also contain zero on the surface)
                seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
                ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

                # Get the number of pixels in the reference surface by counting all pixels that are 1.
                statistics_image_filter.Execute(segmented_surface)
                num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

                # Get all non-zero distances and then add zero distances if required.
                seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
                seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
                seg2ref_distances = seg2ref_distances + list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
                ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
                ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
                ref2seg_distances = ref2seg_distances + list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

                all_surface_distances = seg2ref_distances + ref2seg_distances
                # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In
                # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two
                # segmentations, though in our case it is. More on this below.
                self.surface_distance_results[i, self.surface_distance_measures['mean_surface_distance']] = np.mean(all_surface_distances)
                self.surface_distance_results[i, self.surface_distance_measures['median_surface_distance']] = np.median(all_surface_distances)
                self.surface_distance_results[i, self.surface_distance_measures['std_surface_distance']] = np.std(all_surface_distances)
                self.surface_distance_results[i, self.surface_distance_measures['max_surface_distance']] = np.max(all_surface_distances)


def check_eval_metrics(excel_filename: str):
    """ Just for checking the evaluation metrics"""
    image_size = [64, 64, 100]
    circle_center = [30, 30, 30]
    circle_radius = [20, 20, 20]

    # A filled circle with radius R
    ground_truth = sitk.GaussianSource(sitk.sitkUInt8, image_size, circle_radius, circle_center) > 200
    # A torus with inner radius r
    reference_segmentation1 = ground_truth - (
                sitk.GaussianSource(sitk.sitkUInt8, image_size, circle_radius, circle_center) > 240)
    # A torus with inner radius r_2<r
    reference_segmentation2 = ground_truth - (
                sitk.GaussianSource(sitk.sitkUInt8, image_size, circle_radius, circle_center) > 250)
    ind_T = 30
    #   ['reference 1', 'reference 2', 'segmentation'], figure_size=(12,4));
    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True)
    axs[0].imshow(sitk.GetArrayViewFromImage(ground_truth)[:, :, ind_T])
    axs[0].set_title('ground_truth', fontsize=10)
    axs[1].imshow(sitk.GetArrayViewFromImage(reference_segmentation1)[:, :, ind_T])
    axs[1].set_title('reference_segmentation1', fontsize=10)
    axs[2].imshow(sitk.GetArrayViewFromImage(reference_segmentation2)[:, :, ind_T])
    axs[2].set_title('reference_segmentation2', fontsize=10)
    fig.show()
    segmentations = [reference_segmentation1, reference_segmentation2]
    eval_met = EvaluationMetrics(ground_truth, segmentations)
    eval_met.calculate_statistics()
    eval_met.flush()
    overlap_results = eval_met.overlap_results
    overlap_results_df = pd.DataFrame(data=overlap_results, index=list(range(len(segmentations))),
                                      columns=[name for name in eval_met.overlap_measures.keys()])
    overlap_results_df.plot(kind='bar').legend(bbox_to_anchor=(1.2, 0.9))
    surface_results = eval_met.surface_distance_results
    surface_results_df = pd.DataFrame(data=surface_results, index=list(range(len(segmentations))),
                                      columns=[name for name in eval_met.surface_distance_measures.keys()])
    surface_results_df.plot(kind='bar').legend(bbox_to_anchor=(1.2, 0.9))
    with pd.ExcelWriter(excel_filename) as writer:
        surface_results_df.to_excel(writer, sheet_name='surface_results')
        overlap_results_df.to_excel(writer, sheet_name='overlap_results')

check_eval_metrics(r'C:\Users\alautman\Documents\hjhjh.xlsx')