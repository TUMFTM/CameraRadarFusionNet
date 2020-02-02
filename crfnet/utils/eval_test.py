import numpy as np
from crfnet.utils.eval import evaluate

def evaluate_test_set(model, generator, cfg, mode='test', tensorboard=None, verbose=1):

    # run evaluation
    if cfg.distance_detection:
        best_map, best_st, best_aps, best_precisions, best_recalls, best_mean_loss_errors, best_mean_loss_errors_rel = evaluate(
            generator,
            model,
            distance=cfg.distance_detection,
            iou_threshold=0.5,
            score_threshold=0.05,
            max_detections=100,
            save_path=None,
            render=False,
            workers=cfg.workers
        )

    else:
            best_map, best_st, best_aps, best_precisions, best_recalls  = evaluate(
            generator,
            model,
            distance=cfg.distance_detection,
            iou_threshold=0.5,
            score_threshold=0.05,
            max_detections=100,
            save_path=None,
            render=False,
            workers=cfg.workers
        )
    # ignore bg with [:-1]
    mean_precision = np.nanmean(best_precisions[:-1])
    mean_recall = np.nanmean(best_recalls[:-1])


    # compute per class average precision
    total_instances = []
    precisions = []
    mean_loss_error = np.nan
    mean_loss_error_rel = np.nan
    for label, (average_precision, num_annotations ) in best_aps.items():
        if verbose == 1:
            if cfg.distance_detection:
                print('{:.0f} instances of class'.format(num_annotations),
                    generator.label_to_name(label), 'with average precision: {0:.4f} (precision: {1:.4f}, recall: {2:.4f}) and mean distance error:{3:.2f} or {4:.2f}%'\
                        .format(average_precision, best_precisions[label], best_recalls[label],best_mean_loss_errors[label], best_mean_loss_errors_rel[label]*100))
            else:
                print('{:.0f} instances of class'.format(num_annotations),
                    generator.label_to_name(label), 'with average precision: {0:.4f} (precision: {1:.4f}, recall: {2:.4f})'\
                        .format(average_precision, best_precisions[label], best_recalls[label]))
        total_instances.append(num_annotations)
        precisions.append(average_precision)

    if cfg.weighted_map:
        mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
        if cfg.distance_detection and np.count_nonzero(~np.isnan(best_mean_loss_errors)):
            mean_loss_error = sum([a * b for a, b in zip(total_instances, best_mean_loss_errors) if (b==b)]) / sum(~np.isnan(best_mean_loss_errors)*total_instances)
            mean_loss_error_rel = sum([a * b for a, b in zip(total_instances, best_mean_loss_errors_rel) if (b==b)]) / sum(~np.isnan(best_mean_loss_errors_rel)*total_instances)
    else:
        mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)
        if cfg.distance_detection:
            mean_loss_error = np.nanmean(best_mean_loss_errors)
            mean_loss_error_rel = np.nanmean(best_mean_loss_errors_rel)


    if tensorboard is not None and tensorboard.writer is not None:
        import tensorflow as tf
        summary = tf.Summary()
        summary_map = summary.value.add()
        summary_map.simple_value = mean_ap
        summary_map.tag = "mAP_test_" + mode
        summary_precision = summary.value.add()
        summary_precision.simple_value = mean_precision
        summary_precision.tag = 'precision_test_' + mode
        summary_recall = summary.value.add()
        summary_recall.simple_value = mean_recall
        summary_recall.tag = 'recall_test_' + mode
        if cfg.distance_detection:
            summary_dist = summary.value.add()
            summary_dist.simple_value = mean_loss_error
            summary_dist.tag = 'mADE_test_' + mode
            summary_dist_rel = summary.value.add()
            summary_dist_rel.simple_value = mean_loss_error_rel
            summary_dist_rel.tag = 'mRDE_test_' + mode

        ### class average precisions to tensorboard
        summary_aps = []
        for label, (average_precision, num_annotations ) in best_aps.items():
            if generator.label_to_name(label) is not 'bg':
                summary_aps.append(summary.value.add())
                summary_aps[label].simple_value = average_precision
                summary_aps[label].tag = 'ap_test_' + mode + '_' + generator.label_to_name(label) + ' (' + str(int(num_annotations)) + ' instances)'

        tensorboard.writer.reopen()
        tensorboard.writer.add_summary(summary, 0)
        tensorboard.writer.close()


    if verbose == 1:
        print('='*60)
        print('mAP_test: {0:.4f} \t precision_test:{1:.4f} \t recall_test:{2:.4f}'.format(mean_ap, mean_precision, mean_recall))
        if cfg.distance_detection: print('mADE_test: {0:.2f} \t mRDE_test:{1:.2f}'.format(mean_loss_error, mean_loss_error_rel))
        print('@scorethreshold {0:.2f}'.format(best_st))