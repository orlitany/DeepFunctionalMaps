import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def pointwise_corr_layer(C_est, model_evecs, part_evecs_trans, dist_map):
    """Point-wise correlation between learned descriptors.

    Args:
        C_est: estimated C matrix from previous layer.
        model_evecs: eigen vectors of model shape.
        part_evecs_trans: eigen vectors of part shape, transposed with area preservation factor.
        dist_map: matrix of geodesic distances on model.

    """
    P = tf.matmul(tf.matmul(model_evecs, C_est), part_evecs_trans)
    P = tf.abs(P)

    P_norm = tf.nn.l2_normalize(P, dim=1, name='soft_correspondences')  # normalize the columns

    one_hot = tf.one_hot(indices=tf.argmax(P_norm, 1),
                         depth=tf.shape(P_norm)[1],
                         on_value=True,
                         off_value=False,
                         axis=-1,
                         name='hard_correspondences'
                         )
    one_hot = tf.boolean_mask(dist_map, one_hot)
    average_error = tf.reduce_mean(one_hot)

    loss = tf.nn.l2_loss(dist_map * P_norm)
    loss /= tf.to_float(tf.shape(P)[1] * tf.shape(P)[0])

    return average_error, P_norm, loss


def res_layer(x_in, dims_out, scope, phase):
    """A residual layer implementation.

    Args:
        x_in: input descriptor per point (dims = batch_size X #pts X #channels)
        dims_out: num channles in output. Usually the same as input for a standard resnet layer.
        scope: scope name for variable sharing.
        phase: train\test.

    """
    with tf.variable_scope(scope):
        x = tf.contrib.layers.fully_connected(x_in, dims_out, activation_fn=None, scope='dense_1')
        x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase, scope='bn_1')
        x = tf.nn.relu(x, 'relu')
        x = tf.contrib.layers.fully_connected(x, dims_out, activation_fn=None, scope='dense_2')
        x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase, scope='bn_2')

        # if dims_out change, modify input via linear projection (as suggested in resNet)
        if not x_in.get_shape().as_list()[-1] == dims_out:
            x_in = tf.contrib.layers.fully_connected(x_in, dims_out, activation_fn=None, scope='projection')

        x += x_in

        return tf.nn.relu(x)


def solve_ls(A, B):
    """functional maps layer.

    Args:
        A: part descriptors projected onto part shape eigenvectors.
        B: model descriptors projected onto model shape eigenvectors.

    Returns:
        Ct_est: estimated C (transposed), such that CA ~= B
        safeguard_inverse:
    """

    # transpose input matrices
    At = tf.transpose(A, [0, 2, 1])
    Bt = tf.transpose(B, [0, 2, 1])

    # solve C via least-squares
    Ct_est = tf.matrix_solve_ls(At, Bt)
    C_est = tf.transpose(Ct_est, [0, 2, 1])

    # calculate error for safeguarding
    safeguard_inverse = tf.nn.l2_loss(tf.matmul(At,Ct_est) - Bt) / tf.to_float(tf.reduce_prod(tf.shape(A)))

    return C_est, safeguard_inverse