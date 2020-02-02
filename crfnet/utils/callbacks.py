import keras.callbacks
import keras
import numpy as np
from crfnet.utils.eval import evaluate


class RedirectModel(keras.callbacks.Callback):
    """Callback which wraps another callback, but executed on a different model.

    ```python
    model = keras.models.load_model('model.h5')
    model_checkpoint = ModelCheckpoint(filepath='snapshot.h5')
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.fit(X_train, Y_train, callbacks=[RedirectModel(model_checkpoint, model)])
    ```

    Args
        callback : callback to wrap.
        model    : model to use when executing callbacks.
    """

    def __init__(self,
                 callback,
                 model):
        super(RedirectModel, self).__init__()

        self.callback = callback
        self.redirect_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        # overwrite the model with our custom model
        self.callback.set_model(self.redirect_model)

        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)



class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(
        self,
        generator,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        save_path=None,
        tensorboard=None,
        weighted_average=False,
        verbose=1,
        render=False,
        distance=False,
        workers=0
    ):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            generator        : The generator that represents the dataset to evaluate.
            iou_threshold    : The threshold used to consider when a detection is positive or negative.
            score_threshold  : The score confidence threshold to use for detections.
            max_detections   : The maximum number of detections to use per image.
            save_path        : The path to save images with visualized detections to.
            tensorboard      : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            weighted_average : Compute the mAP using the weighted average of precisions among classes.
            verbose          : Set the verbosity level, by default this is set to 1.
        """
        self.generator       = generator
        self.iou_threshold   = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections  = max_detections
        self.save_path       = save_path
        self.tensorboard     = tensorboard
        self.weighted_average = weighted_average
        self.verbose         = verbose
        self.render          = render
        self.distance        = distance
        self.workers = workers

        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # run evaluation
        if self.distance:
            best_map, best_st, best_aps, best_precisions, best_recalls, best_mean_loss_errors, best_mean_loss_errors_rel = evaluate(
                self.generator,
                self.model,
                distance=self.distance,
                iou_threshold=self.iou_threshold,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
                save_path=self.save_path,
                render=self.render,
                workers=self.workers
            )

        else:
             best_map, best_st, best_aps, best_precisions, best_recalls  = evaluate(
                self.generator,
                self.model,
                distance=self.distance,
                iou_threshold=self.iou_threshold,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
                save_path=self.save_path,
                render=self.render,
                workers=self.workers
            )
        # ignore bg with [:-1]
        self.mean_precision = np.nanmean(best_precisions[:-1])
        self.mean_recall = np.nanmean(best_recalls[:-1])


        # compute per class average precision
        total_instances = []
        precisions = []
        self.mean_loss_error = np.nan
        self.mean_loss_error_rel = np.nan
        for label, (average_precision, num_annotations ) in best_aps.items():
            if self.verbose == 1:
                if self.distance:
                    print('{:.0f} instances of class'.format(num_annotations),
                        self.generator.label_to_name(label), 'with average precision: {0:.4f} (precision: {1:.4f}, recall: {2:.4f}) and mean distance error:{3:.2f} or {4:.2f}%'\
                            .format(average_precision, best_precisions[label], best_recalls[label],best_mean_loss_errors[label], best_mean_loss_errors_rel[label]*100))
                else:
                    print('{:.0f} instances of class'.format(num_annotations),
                        self.generator.label_to_name(label), 'with average precision: {0:.4f} (precision: {1:.4f}, recall: {2:.4f})'\
                            .format(average_precision, best_precisions[label], best_recalls[label]))
            total_instances.append(num_annotations)
            precisions.append(average_precision)

        if self.weighted_average:
            self.mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
            if self.distance and np.count_nonzero(~np.isnan(best_mean_loss_errors)):
                self.mean_loss_error = sum([a * b for a, b in zip(total_instances, best_mean_loss_errors) if (b==b)]) / sum(~np.isnan(best_mean_loss_errors)*total_instances)
                self.mean_loss_error_rel = sum([a * b for a, b in zip(total_instances, best_mean_loss_errors_rel) if (b==b)]) / sum(~np.isnan(best_mean_loss_errors_rel)*total_instances)
        else:
            self.mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)
            if self.distance:
                self.mean_loss_error = np.nanmean(best_mean_loss_errors)
                self.mean_loss_error_rel = np.nanmean(best_mean_loss_errors_rel)


        if self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()
            summary_map = summary.value.add()
            summary_map.simple_value = self.mean_ap
            summary_map.tag = "mAP_val"
            summary_precision = summary.value.add()
            summary_precision.simple_value = self.mean_precision
            summary_precision.tag = 'precision_val'
            summary_recall = summary.value.add()
            summary_recall.simple_value = self.mean_recall
            summary_recall.tag = 'recall_val'
            if self.distance:
                summary_dist = summary.value.add()
                summary_dist.simple_value = self.mean_loss_error
                summary_dist.tag = 'mADE_val'
                summary_dist_rel = summary.value.add()
                summary_dist_rel.simple_value = self.mean_loss_error_rel
                summary_dist_rel.tag = 'mRDE_val'

            ### class average precisions to tensorboard
            summary_aps = []
            for label, (average_precision, num_annotations ) in best_aps.items():
                if self.generator.label_to_name(label) is not 'bg':
                    summary_aps.append(summary.value.add())
                    summary_aps[label].simple_value = average_precision
                    summary_aps[label].tag = 'ap_val_' + self.generator.label_to_name(label) + ' (' + str(int(num_annotations)) + ' instances)'

            self.tensorboard.writer.add_summary(summary, epoch)

        logs['mAP'] = self.mean_ap

        if self.verbose == 1:
            print('='*60)
            print('mAP: {0:.4f} \t precision:{1:.4f} \t recall:{2:.4f}'.format(self.mean_ap, self.mean_precision, self.mean_recall))
            if self.distance: print('mADE: {0:.2f} \t mRDE:{1:.2f}'.format(self.mean_loss_error, self.mean_loss_error_rel))
            print('@scorethreshold {0:.2f}'.format(best_st))
