

##### NEURAL FILTER LOSSES #####
import tensorflow as tf

def binary_focal_loss(loss_fn, threshold=0.5, alpha=0.2, gamma=2.0):
    """
    Compared to focal loss, the binary focal loss

    :param alpha: Scale the focal weight with alpha.
    :param gamma: Take the power of the focal weight with gamma.

    """

    def _binary_focal_loss(y_true, y_pred):

        # apply threshold to get clearly positive and negative predictions
        y_true_binary = tf.keras.backend.greater(y_true, threshold)

        # compute the focal loss
        alpha_factor = tf.keras.backend.ones_like(y_true, dtype=tf.float32) * alpha # create an array with alpha values, same shape as y_true
        alpha_factor = tf.where(y_true_binary, alpha_factor, 1 - alpha_factor) # alpha on true, 1-alpha on false
        alpha_factor = alpha_factor * 2 # we don't want to half the learning rate

        focal_weight = tf.where(y_true_binary, 1 - y_pred, y_pred)

        # this is needed, because the output contains 0.0 after applying to the input grid
        focal_weight = tf.clip_by_value(focal_weight, tf.keras.backend.epsilon(), 1.0) 

        focal_weight = alpha_factor * focal_weight**gamma
        focal_weight = tf.squeeze(focal_weight, axis=-1)
        focal_weight = tf.identity(focal_weight, name="focal_weight")

        cls_loss = focal_weight * loss_fn(y_true, y_pred)
        cls_loss = tf.identity(cls_loss, name="cls_loss")

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(y_true_binary)
        normalizer = tf.keras.backend.cast(tf.keras.backend.shape(normalizer)[0], tf.keras.backend.floatx())
        normalizer = tf.keras.backend.maximum(tf.keras.backend.cast_to_floatx(1), normalizer)

        cls_loss_sum = tf.keras.backend.sum(cls_loss)
        loss = cls_loss_sum / normalizer

        loss = tf.identity(loss, name="focal_loss")
        return loss #tf.keras.backend.sum(cls_loss) / normalizer

    return _binary_focal_loss


def roc_auc_score(y_true, y_pred):
    """ ROC AUC Score.
    Source: https://github.com/tflearn/tflearn/blob/master/tflearn/objectives.py
    Modifications: argument order y_pred and y_true

    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.
    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
    Measures overall performance for a full range of threshold levels.
    Arguments:
        y_true: `Tensor` . Targets (labels), a probability distribution.
        y_pred: `Tensor`. Predicted values.
    """
    with tf.name_scope("RocAucScore"):

        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)

        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.2
        p     = 3

        difference = tf.zeros_like(pos * neg) + pos - neg - gamma

        masked = tf.boolean_mask(difference, difference < 0.0)

        return tf.reduce_sum(tf.pow(-masked, p))

