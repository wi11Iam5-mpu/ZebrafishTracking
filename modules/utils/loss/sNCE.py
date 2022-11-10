def contrastive_loss(features,
                     labels=None,
                     temperature=1.0,
                     contrast_mode=enums.LossContrastMode.ALL_VIEWS,
                     summation_location=enums.LossSummationLocation.OUTSIDE,
                     denominator_mode=enums.LossDenominatorMode.ALL,
                     positives_cap=-1,
                     scale_by_temperature=True):
    r"""Contrastive loss over features.

    Implemented as described in: https://arxiv.org/abs/2004.11362, Equation 2.

    Given `num_views` different views of each of `batch_size` samples, let `f_i`
    (i \in [1, 2 ... (num_views * batch_size)]) denote each respective feature
    vector. The contrastive loss then takes the following form:
  
      L = \sum_{i} L_i

    where each L_i is computed as:

      L_i = -\tau * \sum_{k \in P(i)} \log(p_{ik})    (1)

    where P(i) is the set of positives for entry i (distinct from i) and where:

                         \exp(f_i^T f_k / \tau)
      p_{ik} = ----------------------------------------                        (2)
               \sum_{j \in A(i)} \exp(f_i^T f_j / \tau)

    where A(i) is the set of all positives or negatives (distinct from i). `i` is
    the anchor, and \tau is the temperature.

    This maximizes the likelihood of a given (anchor, positive) pair with
    respect to all possible pairs where the first member is the anchor and the
    second member is a positive or a negative.

    A typical way to define a positive is to define samples from the
    same class (but not the anchor itself) regardless of what view they are from.
    Similarly, a typical way to define a negative is for it to be any view of a
    sample from a different class.

    There are two ways to define which feature pairs should be treated as
    positives and negatives. All views of the same sample are always treated as
    positives. You can declare other samples to be positives by providing `labels`
    such that all samples with the same label will be positives for each other.

    If `labels` is not provided then we default to every sample belonging to its
    own unique class. Therefore, the only positive used is another view of the
    anchor itself. This implements the loss as described in:

      https://arxiv.org/pdf/2002.05709.pdf
      A Simple Framework for Contrastive Learning of Visual Representations
      Chen T., Kornblith S., Norouzi M., Hinton G.

    It is recommended to use features whose L_2 norm is 1. since that ensures
    that the loss does not return NaN values without changing the intended
    behaviour of the loss function.

    In (1) above, note that the summation over positives is located outside of the
    \log(). However, one can permute these two operations. The result is Eq. 3 in
    https://arxiv.org/abs/2004.11362. Users can specify the location of the
    summation relative to the \log() via the `summation_location' argmument:
     - 'out': Eq. 2 in https://arxiv.org/abs/2004.11362.
     - 'in' : Eq. 3 in https://arxiv.org/abs/2004.11362.

    Additionally, in (2) above, note that the denominator sums over *all* entries
    distinct from i. One can change which terms are included in the denominator
    via the `denominator_mode` argument:
     - LossDenominatorMode.ALL : All entries (i.e., all negatives and all
               positives) distinct from i are included.
     - LossDenominatorMode.ONE_POSITIVE : All negatives are included but only the
               single positive in the numerator of (2) is included. Any other
               positives are excluded.
     - LossDenominatorMode.ONLY_NEGATIVES: All negatives are included but no
               positives are, not even the single positive in the numerator of
               (2).

    On TPUs, this method will internally perform the cross-replica operations that
    enable using the samples from all cores in computing the loss. The inputs to
    this function should be the features and labels from a single core and each
    core will compute the loss using just these features as anchors, but will use
    positives and negatives from the full global batch. Since the loss for each
    anchor is only computed on one TPU core, it's still necessary to have a
    cross-replica reduction in the final loss computation.

    Also, though it is not applicable to multiview contrastive learning, this
    function will work if |features| contains only 1 view. In the high batch size
    limit, the implemented contrastive loss with only 1 view, positives_cap = 1,
    and temperature = 1.0 is equivalent to the N-pairs loss
    (https://papers.nips.cc/paper/6200-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective.pdf)

    Args:
      features: A Tensor of rank at least 3, where the first 2 dimensions are
        batch_size and num_views, and the remaining dimensions are the feature
        shape. Note that when running on TPU, batch_size is the per-core batch
        size.
      labels: One-hot labels to be used to construct the supervised contrastive
        loss. Samples with the same labels are used as positives for each other.
        Labels must have shape [batch_size, num_labels] with numeric dtype and be
        0-1 valued. Note that when running on TPU, batch_size is the per-core
        batch size.
      temperature: Temperature at which softmax evaluation is done. Temperature
        must be a python scalar or scalar Tensor of numeric dtype.
      contrast_mode: LossContrastMode specifying which views get used as anchors
        (f_i in the expression above)
        'ALL_VIEWS': All the views of all samples are used as anchors (f_i in the
          expression above).
        'ONE_VIEW': Just the first view of each sample is used as an anchor (f_i
          in the expression above). This view is called the `core` view against
          which other views are contrasted.
      summation_location: LossSummationLocation specifying location of positives
        summation. See documentation above for more details.
      denominator_mode: LossDenominatorMode specifying which positives to include
        in contrastive denominator. See documentation above for more details.
      positives_cap: Integer maximum number of positives *other* than
        augmentations of anchor. Infinite if < 0. Must be multiple of num_views.
        Including augmentations, a maximum of (positives_cap + num_views - 1)
        positives is possible. This parameter modifies the contrastive numerator
        by selecting which positives are present in the summation, and which
        positives contribure to the denominator if denominator_mode ==
        enums.LossDenominatorMode.ALL.
      scale_by_temperature: Boolean. Whether to scale the loss by `temperature`.
        The loss gradient naturally has a 1/temperature scaling factor, so this
        counteracts it.

    Returns:
      Scalar tensor with contrastive loss value with shape [batch_size] and dtype
      tf.float32. The loss for each batch element is the mean over all views.

    Raises:
      ValueError if the shapes of any of the Tensors are unexpected, or if both
      `labels` and `mask` are not `None`.
    """
    features = features if features is not None else None
    labels = labels if labels is not None else None

    # _validate_contrastive_loss_inputs
    local_batch_size, num_views = check(
        features, labels, contrast_mode, summation_location, denominator_mode,
        positives_cap)

    # Flatten `features` to a single dimension per view per sample so it has shape
    # [local_batch_size, num_views, num_features].
    if features.shape.rank > 3:
        features = tf.reshape(features,
                              tf.concat([tf.shape(features)[:2], [-1]], axis=0),
                              'flattened_features')
    if features.dtype != tf.float32:
        features = tf.cast(features, tf.float32)

    # Grab the features from all TPU cores. We use the local batch as anchors and
    # the full global batch as contrastives. If not on TPU, global_features is the
    # same as features.
    global_features = utils.cross_replica_concat(features)
    global_batch_size = tf.compat.dimension_at_index(global_features.shape,
                                                     0).value
    local_replica_id = utils.local_tpu_replica_id()

    # Generate the [local_batch_size, global_batch_size] slice of the
    # [global_batch_size, global_batch_size] identity matrix that corresponds to
    # the current replica.
    diagonal_mask = tf.one_hot(
        tf.range(local_batch_size) + (local_replica_id * local_batch_size),
        global_batch_size)

    # Generate `mask` with shape [local_batch_size, global_batch_size] that
    # indicates which samples should be considered positives for each other.
    if labels is None:
        # Defaults to every sample belonging to its own unique class, containing
        # just that sample and other views of it.
        mask = diagonal_mask
    else:
        labels = tf.cast(labels, tf.float32)  # TPU matmul op unsupported for ints.
        global_labels = utils.cross_replica_concat(labels)
        mask = tf.linalg.matmul(labels, global_labels, transpose_b=True)
    mask = tf.ensure_shape(mask, [local_batch_size, global_batch_size])

    # To streamline the subsequent TF, the first two dimensions of
    # `global_features` (i.e., global_batch_size and num_views) should be
    # transposed and then flattened. The result has shape
    # [num_views * global_batch_size, num_features], and its first dimension
    # elements are grouped by view, not by sample.
    all_global_features = tf.reshape(
        tf.transpose(global_features, perm=[1, 0, 2]),
        [num_views * global_batch_size, -1])

    if contrast_mode == enums.LossContrastMode.ONE_VIEW:
        anchor_features = features[:, 0]
        num_anchor_views = 1
    else:  # contrast_mode == enums.LossContrastMode.ALL_VIEWS
        # Reshape features to match how global_features is reshaped above.
        anchor_features = tf.reshape(
            tf.transpose(features, perm=[1, 0, 2]),
            [num_views * local_batch_size, -1])
        num_anchor_views = num_views

    # Generate `logits`, the tensor of (temperature-scaled) dot products of the
    # anchor features with all features. It has shape
    # [local_batch_size * num_anchor_views, global_batch_size * num_views]. To
    # improve numerical stability, subtract out the largest |logits| element in
    # each row from all elements in that row. Since |logits| is only ever used as
    # a ratio of exponentials of |logits| values, this subtraction does not change
    # the results correctness. A stop_gradient() is needed because this change is
    # just for numerical precision.
    logits = tf.linalg.matmul(
        anchor_features, all_global_features, transpose_b=True)
    temperature = tf.cast(temperature, tf.float32)
    logits = logits / temperature
    logits = (
            logits - tf.reduce_max(tf.stop_gradient(logits), axis=1, keepdims=True))
    exp_logits = tf.exp(logits)

    # The following masks are all tiled by the number of views, i.e., they have
    # shape [local_batch_size * num_anchor_views, global_batch_size * num_views].
    positives_mask, negatives_mask = (
        _create_tiled_masks(mask, diagonal_mask, num_views, num_anchor_views,
                            positives_cap))
    num_positives_per_row = tf.reduce_sum(positives_mask, axis=1)

    if denominator_mode == enums.LossDenominatorMode.ALL:
        denominator = tf.reduce_sum(
            exp_logits * negatives_mask, axis=1, keepdims=True) + tf.reduce_sum(
            exp_logits * positives_mask, axis=1, keepdims=True)
    elif denominator_mode == enums.LossDenominatorMode.ONE_POSITIVE:
        denominator = exp_logits + tf.reduce_sum(
            exp_logits * negatives_mask, axis=1, keepdims=True)
    else:  # denominator_mode == enums.LossDenominatorMode.ONLY_NEGATIVES
        denominator = tf.reduce_sum(
            exp_logits * negatives_mask, axis=1, keepdims=True)

    # Note that num_positives_per_row can be zero only if 1 view is used. The
    # various tf.math.divide_no_nan() calls below are to handle this case.
    if summation_location == enums.LossSummationLocation.OUTSIDE:
        log_probs = (logits - tf.math.log(denominator)) * positives_mask
        log_probs = tf.reduce_sum(log_probs, axis=1)
        log_probs = tf.math.divide_no_nan(log_probs, num_positives_per_row)
    else:  # summation_location == enums.LossSummationLocation.INSIDE
        log_probs = exp_logits / denominator * positives_mask
        log_probs = tf.reduce_sum(log_probs, axis=1)
        log_probs = tf.math.divide_no_nan(log_probs, num_positives_per_row)
        log_probs = tf.math.log(log_probs)

    loss = -log_probs
    if scale_by_temperature:
        loss *= temperature
    loss = tf.reshape(loss, [num_anchor_views, local_batch_size])

    if num_views != 1:
        loss = tf.reduce_mean(loss, axis=0)
    else:
        # The 1 view case requires special handling bc, unlike in the > 1 view case,
        # not all samples are guaranteed to have a positive. Also, no reduction over
        # views is needed.
        num_valid_views_per_sample = (
            tf.reshape(num_positives_per_row, [1, local_batch_size]))
        loss = tf.squeeze(tf.math.divide_no_nan(loss, num_valid_views_per_sample))

    return loss
