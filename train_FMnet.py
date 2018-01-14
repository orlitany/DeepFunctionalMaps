import os
import time
import tensorflow as tf
import numpy as np
import scipy.io as sio

from models import fmnet_model


flags = tf.app.flags
FLAGS = flags.FLAGS

# training params
flags.DEFINE_float('learning_rate', 1e-3, 'initial learning rate.')
flags.DEFINE_integer('batch_size', 32, 'batch size.')
flags.DEFINE_integer('queue_size', 10, '')

# architecture parameters
flags.DEFINE_integer('num_layers', 7, 'network depth')
flags.DEFINE_integer('num_evecs', 120,
					 'number of eigenvectors used for representation. The first 500 are precomputed and stored in input')
flags.DEFINE_integer('dim_shot', 352, '')

# data parameters
flags.DEFINE_string('models_dir', './Data/faust_models/', '')
flags.DEFINE_string('dist_maps', './Data/distance_maps/', '')

flags.DEFINE_string('log_dir', './Results/train_inter_k_flag', 'directory to save models and results')
flags.DEFINE_integer('max_train_iter', 500000, '')
flags.DEFINE_integer('save_summaries_secs', 60, '')
flags.DEFINE_integer('save_model_secs', 1200, '')
flags.DEFINE_string('master', '', '')
flags.DEFINE_integer('run_validation_every', 150, '')
flags.DEFINE_integer('validation_size', 10, '')

# globals
train_subjects = [0, 1, 2, 3, 4, 5, 6, 7]
validation_subjects = [8, 9]
flags.DEFINE_integer('num_poses_per_subject_total', 10, '')
dist_maps = {}


def get_input_pair(batch_size=1, num_vertices=1500):
	dataset = 'train'
	batch_input = {'part_evecs': np.zeros((batch_size, num_vertices, FLAGS.num_evecs)),
				   'model_evecs': np.zeros((batch_size, num_vertices, FLAGS.num_evecs)),
				   'part_evecs_trans': np.zeros((batch_size, FLAGS.num_evecs, num_vertices)),
				   'model_evecs_trans': np.zeros((batch_size, FLAGS.num_evecs, num_vertices)),
				   'part_shot': np.zeros((batch_size, num_vertices, FLAGS.dim_shot)),
				   'model_shot': np.zeros((batch_size, num_vertices, FLAGS.dim_shot))
				   }

	batch_dist = np.zeros((batch_size, num_vertices, num_vertices))

	for i_batch in range(batch_size):
		i_subject1 = np.random.choice(train_subjects)  # model
		i_subject2 = np.random.choice(train_subjects)
		i_model = FLAGS.num_poses_per_subject_total * i_subject1 + \
				  np.random.randint(0, FLAGS.num_poses_per_subject_total, 1)[0]
		i_part = FLAGS.num_poses_per_subject_total * i_subject2 + \
				 np.random.randint(0, FLAGS.num_poses_per_subject_total, 1)[0]

		batch_input_, batch_dist_ = get_pair_from_ram(i_subject1, i_model, i_part, dataset)

		batch_input_['part_labels'] = range(np.shape(batch_input_['part_evecs'])[0])  # replace once we switch to scans
		batch_input_['model_labels'] = range(np.shape(batch_input_['model_evecs'])[0])
		joint_labels = np.intersect1d(batch_input_['part_labels'], batch_input_['model_labels'])
		joint_labels = np.random.permutation(joint_labels)[:num_vertices]

		ind_dict_part = {value: ind for ind, value in enumerate(batch_input_['part_labels'])}
		ind_part = [ind_dict_part[x] for x in joint_labels]

		ind_dict_model = {value: ind for ind, value in enumerate(batch_input_['model_labels'])}
		ind_model = [ind_dict_model[x] for x in joint_labels]

		assert len(ind_part) == len(ind_model), 'number of indices must be equal'

		batch_dist[i_batch] = batch_dist_[joint_labels, :][:, joint_labels]  # slice the common indices
		batch_input['part_evecs'][i_batch] = batch_input_['part_evecs'][ind_part, :]
		batch_input['part_evecs_trans'][i_batch] = batch_input_['part_evecs_trans'][:, ind_part]
		batch_input['part_shot'][i_batch] = batch_input_['part_shot'][ind_part, :]
		batch_input['model_evecs'][i_batch] = batch_input_['model_evecs'][ind_model, :]
		batch_input['model_evecs_trans'][i_batch] = batch_input_['model_evecs_trans'][:, ind_model]
		batch_input['model_shot'][i_batch] = batch_input_['model_shot'][ind_model, :]

	return batch_input, batch_dist


def get_pair_from_ram(i_subject, i_model, i_part, dataset):
	input_data = {}

	if dataset == 'train':
		input_data['part_evecs'] = models_train[i_part]['model_evecs']
		input_data['part_evecs_trans'] = models_train[i_part]['model_evecs_trans']
		input_data['part_shot'] = models_train[i_part]['model_shot']
		input_data.update(models_train[i_model])
	else:
		input_data['part_evecs'] = models_val[i_part]['model_evecs']
		input_data['part_evecs_trans'] = models_val[i_part]['model_evecs_trans']
		input_data['part_shot'] = models_val[i_part]['model_shot']
		input_data.update(models_val[i_model])

	# m_star from dist_map
	m_star = dist_maps[i_subject]

	return input_data, m_star


def load_models_to_ram():
	global models_train
	models_train = {}
	global models_val
	models_val = {}

	# load model, part and labels
	for i_subject in train_subjects:
		for i_model in range(i_subject * FLAGS.num_poses_per_subject_total,
							 FLAGS.num_poses_per_subject_total * (i_subject + 1)):
			model_file = FLAGS.models_dir + '/' + 'train' + '/' + 'tr_reg_%.3d.mat' % i_model
			input_data = sio.loadmat(model_file)
			input_data['model_evecs'] = input_data['model_evecs'][:, 0:FLAGS.num_evecs]
			input_data['model_evecs_trans'] = input_data['model_evecs_trans'][0:FLAGS.num_evecs, :]
			models_train[i_model] = input_data

	for i_subject in validation_subjects:
		for i_model in range(i_subject * FLAGS.num_poses_per_subject_total,
							 FLAGS.num_poses_per_subject_total * (i_subject + 1)):
			model_file = FLAGS.models_dir + '/' + 'test' + '/' + 'tr_reg_%.3d.mat' % i_model
			input_data = sio.loadmat(model_file)
			input_data['model_evecs'] = input_data['model_evecs'][:, 0:FLAGS.num_evecs]
			input_data['model_evecs_trans'] = input_data['model_evecs_trans'][0:FLAGS.num_evecs, :]
			models_val[i_model] = input_data


def load_dist_maps():
	print('loading dist maps...')
	# load distance maps to memory for both training and validation sets
	for i_subject in train_subjects + validation_subjects:
		global dist_maps
		d = sio.loadmat(FLAGS.dist_maps + 'tr_reg_%.3d.mat' % (i_subject * FLAGS.num_poses_per_subject_total))
		dist_maps[i_subject] = d['D']


def run_training():

	print('log_dir=%s' % FLAGS.log_dir)
	if not os.path.isdir(FLAGS.log_dir):
		os.makedirs(FLAGS.log_dir)
	print('num_evecs=%d' % FLAGS.num_evecs)

	print('building graph...')
	with tf.Graph().as_default():

		# Set placeholders for inputs
		part_shot = tf.placeholder(tf.float32, shape=(None, None, FLAGS.dim_shot), name='part_shot')
		model_shot = tf.placeholder(tf.float32, shape=(None, None, FLAGS.dim_shot), name='model_shot')
		dist_map = tf.placeholder(tf.float32, shape=(None, None, None), name='dist_map')
		part_evecs = tf.placeholder(tf.float32, shape= (None, None, FLAGS.num_evecs), name='part_evecs')
		part_evecs_trans = tf.placeholder(tf.float32, shape=(None, FLAGS.num_evecs, None), name='part_evecs_trans')
		model_evecs = tf.placeholder(tf.float32, shape= (None, None, FLAGS.num_evecs), name='model_evecs')
		model_evecs_trans = tf.placeholder(tf.float32, shape=(None, FLAGS.num_evecs, None), name='model_evecs_trans')

		# train\test switch flag
		phase = tf.placeholder(dtype=tf.bool, name='phase')

		net_loss, safeguard_inverse, merged, P_norm, average_error, net = fmnet_model(phase, part_shot, model_shot,
																					  dist_map, part_evecs,
																					  part_evecs_trans, model_evecs,
																					  model_evecs_trans)
		summary = tf.summary.scalar("num_evecs", float(FLAGS.num_evecs))

		global_step = tf.Variable(0, name='global_step', trainable=False)

		optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

		train_op = optimizer.minimize(net_loss, global_step=global_step)

		saver = tf.train.Saver(max_to_keep=100)
		sv = tf.train.Supervisor(logdir=FLAGS.log_dir,
								 init_op=tf.global_variables_initializer(),
								 local_init_op=tf.local_variables_initializer(),
								 global_step=global_step,
								 save_summaries_secs=FLAGS.save_summaries_secs,
								 save_model_secs=FLAGS.save_model_secs,
								 summary_op=None,
								 saver=saver)

		writer = sv.summary_writer


		print('starting session...')
		iteration = 0
		with sv.managed_session(master=FLAGS.master) as sess:

			print('loading data to ram...')
			load_models_to_ram()

			load_dist_maps()

			print('starting training loop...')
			while not sv.should_stop() and iteration < FLAGS.max_train_iter:
				iteration += 1
				start_time = time.time()

				input_data, mstar = get_input_pair(FLAGS.batch_size)

				feed_dict = {phase: True,
							 part_shot: input_data['part_shot'],
							 model_shot: input_data['model_shot'],
							 dist_map: mstar,
							 part_evecs: input_data['part_evecs'],
							 part_evecs_trans: input_data['part_evecs_trans'],
							 model_evecs: input_data['model_evecs'],
							 model_evecs_trans: input_data['model_evecs_trans'],
							 }

				summaries, step, my_loss, safeguard, _ = sess.run(
					[merged, global_step, net_loss, safeguard_inverse, train_op], feed_dict=feed_dict)
				writer.add_summary(summaries, step)
				summary_ = sess.run(summary)
				writer.add_summary(summary_, step)

				duration = time.time() - start_time

				print('train - step %d: loss = %.4f (%.3f sec)' % (step, my_loss, duration))


			saver.save(sess, FLAGS.log_dir + '/model.ckpt', global_step=step)
			writer.flush()
			sv.request_stop()
			sv.stop()


def main(_):
	run_training()


if __name__ == '__main__':
	tf.app.run()