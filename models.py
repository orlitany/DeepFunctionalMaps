import tensorflow as tf
import numpy as np

from ops import *

flags = tf.app.flags
FLAGS = flags.FLAGS


def fmnet_model(phase, part_shot, model_shot, dist_map, part_evecs,	part_evecs_trans, model_evecs, model_evecs_trans):
	"""Build FM-net model.

	Args:
		phase: train\test.
		part_shot: SHOT descriptor of source shape (part).
		model_shot: SHOT descriptor of target shape (model).
		dist_map: distance map on target shape to evaluate geodesic error
		part_evecs: eigenvectors on source shape
		part_evecs_trans: transposed part_evecs with mass matrix correction
		model_evecs: eigenvectors on target shape
		model_evecs_trans: transposed model_evecs with mass matrix correction

	"""

	net = {}

	for i_layer in range(FLAGS.num_layers):
		with tf.variable_scope("layer_%d" % i_layer) as scope:
			if i_layer == 0:
				net['layer_%d_part' % i_layer] = res_layer(part_shot, dims_out=int(part_shot.shape[-1]), scope=scope,
														   phase=phase)
				scope.reuse_variables()
				net['layer_%d_model' % i_layer] = res_layer(model_shot, dims_out=int(model_shot.shape[-1]), scope=scope,
															phase=phase)
			else:
				net['layer_%d_part' % i_layer] = res_layer(net['layer_%d_part' % (i_layer - 1)],
														   dims_out=int(part_shot.shape[-1]),
														   scope=scope, phase=phase)
				scope.reuse_variables()
				net['layer_%d_model' % i_layer] = res_layer(net['layer_%d_model' % (i_layer - 1)],
															dims_out=int(part_shot.shape[-1]),
															scope=scope, phase=phase)

	#  project output features on the shape Laplacian eigen functions
	layer_C_est = i_layer + 1  # grab current layer index
	A = tf.matmul(part_evecs_trans, net['layer_%d_part' % (layer_C_est - 1)])
	net['A'] = A
	B = tf.matmul(model_evecs_trans, net['layer_%d_model' % (layer_C_est - 1)])
	net['B'] = B

	#  FM-layer: evaluate C_est
	net['C_est'], safeguard_inverse = solve_ls(A, B)

	#  Evaluate loss via soft-correspondence error
	with tf.variable_scope("pointwise_corr_loss"):
		average_error, P_norm, net_loss = pointwise_corr_layer(net['C_est'], model_evecs, part_evecs_trans, dist_map)

	tf.summary.scalar('net_loss', net_loss)
	tf.summary.scalar('average_error', average_error)
	merged = tf.summary.merge_all()

	return net_loss, safeguard_inverse, merged, P_norm, average_error, net