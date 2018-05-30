# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A program to annotate a conll file with a tensorflow neural net parser."""


import tensorflow as tf

import graph_builder
import structured_graph_builder
from ops import gen_parser_ops

flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('task_context', '',
                    'Path to a task context with inputs and parameters for '
                    'feature extractors.')
flags.DEFINE_string('resource_dir', '',
                    'Optional base directory for task context resources.')
flags.DEFINE_string('model_path', '', 'Path to model parameters.')
flags.DEFINE_string('arg_prefix', None, 'Prefix for context parameters.')
flags.DEFINE_string('graph_builder', 'greedy',
                    'Which graph builder to use, either greedy or structured.')
flags.DEFINE_string('input', 'stdin',
                    'Name of the context input to read data from.')
flags.DEFINE_string('output', 'stdout-conll',
                    'Name of the context input to write data to.')
flags.DEFINE_string('hidden_layer_sizes', '200,200',
                    'Comma separated list of hidden layer sizes.')
flags.DEFINE_string('nndev', '/cpu:0',
                    'Calculating device for feed-forward neural network.')
flags.DEFINE_float('gpu_mem_frac', 1.0, 'GPU memory fraction')
flags.DEFINE_integer('batch_size', 32,
                     'Number of sentences to process in parallel.')
flags.DEFINE_integer('beam_size', 8, 'Number of slots for beam parsing.')
flags.DEFINE_integer('max_steps', 1000, 'Max number of steps to take.')
flags.DEFINE_bool('slim_model', False,
                  'Whether to expect only averaged variables.')


def Eval(feature_sizes, domain_sizes, embedding_dims, num_actions):
  """Builds network."""

  hidden_layer_sizes = map(int, FLAGS.hidden_layer_sizes.split(','))
  if FLAGS.graph_builder == 'greedy':
    parser = graph_builder.GreedyParser(num_actions,
                                        feature_sizes,
                                        domain_sizes,
                                        embedding_dims,
                                        hidden_layer_sizes,
                                        nndev=FLAGS.nndev,
                                        gate_gradients=True,
                                        arg_prefix=FLAGS.arg_prefix)
  else:
    parser = structured_graph_builder.StructuredGraphBuilder(
        num_actions,
        feature_sizes,
        domain_sizes,
        embedding_dims,
        hidden_layer_sizes,
        nndev=FLAGS.nndev,
        gate_gradients=True,
        arg_prefix=FLAGS.arg_prefix,
        beam_size=FLAGS.beam_size,
        max_steps=FLAGS.max_steps)
  parser.AddEvaluation(FLAGS.task_context,
                       FLAGS.batch_size,
                       corpus_name=FLAGS.input,
                       evaluation_max_steps=FLAGS.max_steps)

  parser.AddSaver(FLAGS.slim_model)

  sink_documents = tf.placeholder(tf.string)
  sink = gen_parser_ops.document_sink(sink_documents,
                                      task_context=FLAGS.task_context,
                                      corpus_name=FLAGS.output)
  return parser, sink, sink_documents


def main(unused_argv):
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_mem_frac)
  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2,
                                        inter_op_parallelism_threads=2,
                                        gpu_options=gpu_options)) as sess:
    feature_sizes, domain_sizes, embedding_dims, num_actions = sess.run(
      gen_parser_ops.feature_size(task_context=FLAGS.task_context,
                                  arg_prefix=FLAGS.arg_prefix))
    parser, sink, sink_documents = Eval(feature_sizes, domain_sizes, embedding_dims, num_actions)
    sess.run(parser.inits.values())
    parser.saver.restore(sess, FLAGS.model_path)

    num_epochs = None
    while True:
      tf_eval_epochs, tf_documents = sess.run([
          parser.evaluation['epochs'],
          parser.evaluation['documents'],
      ])
      if len(tf_documents):
        sess.run(sink, feed_dict={sink_documents: tf_documents})
      if num_epochs is None:
        num_epochs = tf_eval_epochs
      elif num_epochs < tf_eval_epochs:
        break

if __name__ == '__main__':
  tf.app.run()
