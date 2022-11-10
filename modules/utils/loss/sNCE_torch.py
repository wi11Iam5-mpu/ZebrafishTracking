def contrastive_loss(features,
                     labels=None,
                     temperature=1.0,
                     positives_cap=-1,
                     scale_by_temperature=True):

    global_features = features if features is not None else None
    global_batch_size = labels if labels is not None else None

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
    num_anchor_views = num_views

    logits = tf.linalg.matmul(
        anchor_features, all_global_features, transpose_b=True)
    temperature = temperature
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
