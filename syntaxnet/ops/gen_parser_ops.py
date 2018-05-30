"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import op_def_library


_beam_eval_output_outputs = ["eval_metrics", "documents"]


_BeamEvalOutputOutput = collections.namedtuple("BeamEvalOutput",
                                               _beam_eval_output_outputs)


def beam_eval_output(beam_state, name=None):
  r"""Computes eval metrics for the best paths in the input beams.

  Args:
    beam_state: A `Tensor` of type `int64`. beam state handle.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (eval_metrics, documents).
    eval_metrics: A `Tensor` of type `int32`. token counts used to compute evaluation metrics.
    documents: A `Tensor` of type `string`. parsed documents.
  """
  result = _op_def_lib.apply_op("BeamEvalOutput", beam_state=beam_state,
                                name=name)
  return _BeamEvalOutputOutput._make(result)


ops.RegisterShape("BeamEvalOutput")(None)
_beam_parse_reader_outputs = ["features", "beam_state", "num_epochs"]


_BeamParseReaderOutput = collections.namedtuple("BeamParseReader",
                                                _beam_parse_reader_outputs)


def beam_parse_reader(task_context, feature_size, beam_size, batch_size=None,
                      corpus_name=None, allow_feature_weights=None,
                      arg_prefix=None, continue_until_all_final=None,
                      always_start_new_sentences=None, name=None):
  r"""Reads sentences and creates a beam parser.

  Args:
    task_context: A `string`. file path at which to read the task context.
    feature_size: An `int` that is `>= 1`.
      number of feature outputs emitted by this reader.
    beam_size: An `int`. limit on the beam size.
    batch_size: An optional `int`. Defaults to `1`.
    corpus_name: An optional `string`. Defaults to `"documents"`.
      name of task input in the task context to read parses from.
    allow_feature_weights: An optional `bool`. Defaults to `True`.
      whether the op is expected to output weighted features.
      If false, it will check that no weights are specified.
    arg_prefix: An optional `string`. Defaults to `"brain_parser"`.
      prefix for context parameters.
    continue_until_all_final: An optional `bool`. Defaults to `False`.
      whether to continue parsing after the gold path falls
      off the beam.
    always_start_new_sentences: An optional `bool`. Defaults to `False`.
      whether to skip to the beginning of a new sentence
      after each training step.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (features, beam_state, num_epochs).
    features: A list of `feature_size` `Tensor` objects of type `string`. features firing at the initial parser state encoded as
      dist_belief.SparseFeatures protocol buffers.
    beam_state: A `Tensor` of type `int64`. beam state handle.
    num_epochs: A `Tensor` of type `int32`.
  """
  result = _op_def_lib.apply_op("BeamParseReader", task_context=task_context,
                                feature_size=feature_size,
                                beam_size=beam_size, batch_size=batch_size,
                                corpus_name=corpus_name,
                                allow_feature_weights=allow_feature_weights,
                                arg_prefix=arg_prefix,
                                continue_until_all_final=continue_until_all_final,
                                always_start_new_sentences=always_start_new_sentences,
                                name=name)
  return _BeamParseReaderOutput._make(result)


ops.RegisterShape("BeamParseReader")(None)
_beam_parser_outputs = ["features", "next_beam_state", "alive"]


_BeamParserOutput = collections.namedtuple("BeamParser", _beam_parser_outputs)


def beam_parser(beam_state, transition_scores, feature_size, name=None):
  r"""Updates the beam parser based on scores in the input transition scores.

  Args:
    beam_state: A `Tensor` of type `int64`. beam state.
    transition_scores: A `Tensor` of type `float32`.
      scores for every transition from the current parser state.
    feature_size: An `int` that is `>= 1`.
      number of feature outputs emitted by this reader.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (features, next_beam_state, alive).
    features: A list of `feature_size` `Tensor` objects of type `string`. features firing at the current parser state encoded as
      dist_belief.SparseFeatures protocol buffers.
    next_beam_state: A `Tensor` of type `int64`. beam state handle.
    alive: A `Tensor` of type `bool`. whether the gold state is still in the beam.
  """
  result = _op_def_lib.apply_op("BeamParser", beam_state=beam_state,
                                transition_scores=transition_scores,
                                feature_size=feature_size, name=name)
  return _BeamParserOutput._make(result)


ops.RegisterShape("BeamParser")(None)
_beam_parser_output_outputs = ["indices_and_paths", "batches_and_slots",
                              "gold_slot", "path_scores"]


_BeamParserOutputOutput = collections.namedtuple("BeamParserOutput",
                                                 _beam_parser_output_outputs)


def beam_parser_output(beam_state, name=None):
  r"""Converts the current state of the beam parser into a set of indices into

  the scoring matrices that lead there.

  Args:
    beam_state: A `Tensor` of type `int64`. beam state handle.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (indices_and_paths, batches_and_slots, gold_slot, path_scores).
    indices_and_paths: A `Tensor` of type `int32`. matrix whose first row is a vector to look up beam paths and
      decisions with, and whose second row are the corresponding
      path ids.
    batches_and_slots: A `Tensor` of type `int32`. matrix whose first row is a vector identifying the batch to
      which the paths correspond, and whose second row are the
      slots.
    gold_slot: A `Tensor` of type `int32`. location in final beam of the gold path [batch_size].
    path_scores: A `Tensor` of type `float32`. cumulative sum of scores along each path in each beam. Within each
      beam, scores are sorted from low to high.
  """
  result = _op_def_lib.apply_op("BeamParserOutput", beam_state=beam_state,
                                name=name)
  return _BeamParserOutputOutput._make(result)


ops.RegisterShape("BeamParserOutput")(None)
_decoded_parse_reader_outputs = ["features", "num_epochs", "eval_metrics",
                                "documents"]


_DecodedParseReaderOutput = collections.namedtuple("DecodedParseReader",
                                                   _decoded_parse_reader_outputs)


def decoded_parse_reader(transition_scores, task_context, feature_size,
                         batch_size, corpus_name=None, arg_prefix=None,
                         name=None):
  r"""Reads sentences and parses them taking parsing transitions based on the

  input transition scores.

  Args:
    transition_scores: A `Tensor` of type `float32`.
      scores for every transition from the current parser state.
    task_context: A `string`. file path at which to read the task context.
    feature_size: An `int` that is `>= 1`.
      number of feature outputs emitted by this reader.
    batch_size: An `int`. number of sentences to parse at a time.
    corpus_name: An optional `string`. Defaults to `"documents"`.
      name of task input in the task context to read parses from.
    arg_prefix: An optional `string`. Defaults to `"brain_parser"`.
      prefix for context parameters.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (features, num_epochs, eval_metrics, documents).
    features: A list of `feature_size` `Tensor` objects of type `string`. features firing at the current parser state encoded as
      dist_belief.SparseFeatures protocol buffers.
    num_epochs: A `Tensor` of type `int32`. number of times this reader went over the training corpus.
    eval_metrics: A `Tensor` of type `int32`. token counts used to compute evaluation metrics.
    documents: A `Tensor` of type `string`.
  """
  result = _op_def_lib.apply_op("DecodedParseReader",
                                transition_scores=transition_scores,
                                task_context=task_context,
                                feature_size=feature_size,
                                batch_size=batch_size,
                                corpus_name=corpus_name,
                                arg_prefix=arg_prefix, name=name)
  return _DecodedParseReaderOutput._make(result)


ops.RegisterShape("DecodedParseReader")(None)
_document_sink_outputs = [""]


def document_sink(documents, task_context, corpus_name=None, name=None):
  r"""Write documents to documents_path.

  Args:
    documents: A `Tensor` of type `string`. documents to write.
    task_context: A `string`.
    corpus_name: An optional `string`. Defaults to `"documents"`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("DocumentSink", documents=documents,
                                task_context=task_context,
                                corpus_name=corpus_name, name=name)
  return result


ops.RegisterShape("DocumentSink")(None)
_document_source_outputs = ["documents", "last"]


_DocumentSourceOutput = collections.namedtuple("DocumentSource",
                                               _document_source_outputs)


def document_source(task_context, batch_size, corpus_name=None, name=None):
  r"""Reads documents from documents_path and outputs them.

  Args:
    task_context: A `string`.
    batch_size: An `int`. how many documents to read at once.
    corpus_name: An optional `string`. Defaults to `"documents"`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (documents, last).
    documents: A `Tensor` of type `string`. a vector of documents as serialized protos.
    last: A `Tensor` of type `bool`. whether this is the last batch of documents from this document path.
  """
  result = _op_def_lib.apply_op("DocumentSource", task_context=task_context,
                                batch_size=batch_size,
                                corpus_name=corpus_name, name=name)
  return _DocumentSourceOutput._make(result)


ops.RegisterShape("DocumentSource")(None)
_feature_size_outputs = ["feature_sizes", "domain_sizes", "embedding_dims",
                        "num_actions"]


_FeatureSizeOutput = collections.namedtuple("FeatureSize",
                                            _feature_size_outputs)


def feature_size(task_context, arg_prefix=None, name=None):
  r"""An op that returns the number and domain sizes of parser features.

  Args:
    task_context: A `string`. file path at which to read the task context.
    arg_prefix: An optional `string`. Defaults to `"brain_parser"`.
      prefix for context parameters.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (feature_sizes, domain_sizes, embedding_dims, num_actions).
    feature_sizes: A `Tensor` of type `int32`. number of feature locators in each group of parser features.
    domain_sizes: A `Tensor` of type `int32`. domain size for each feature group of parser features.
    embedding_dims: A `Tensor` of type `int32`. embedding dimension for each feature group of parser features.
    num_actions: A `Tensor` of type `int32`. number of actions a parser can perform.
  """
  result = _op_def_lib.apply_op("FeatureSize", task_context=task_context,
                                arg_prefix=arg_prefix, name=name)
  return _FeatureSizeOutput._make(result)


ops.RegisterShape("FeatureSize")(None)
_gold_parse_reader_outputs = ["features", "num_epochs", "gold_actions"]


_GoldParseReaderOutput = collections.namedtuple("GoldParseReader",
                                                _gold_parse_reader_outputs)


def gold_parse_reader(task_context, feature_size, batch_size,
                      corpus_name=None, arg_prefix=None, name=None):
  r"""Reads sentences, parses them, and returns (gold action, feature) pairs.

  Args:
    task_context: A `string`. file path at which to read the task context.
    feature_size: An `int` that is `>= 1`.
      number of feature outputs emitted by this reader.
    batch_size: An `int`. number of sentences to parse at a time.
    corpus_name: An optional `string`. Defaults to `"documents"`.
      name of task input in the task context to read parses from.
    arg_prefix: An optional `string`. Defaults to `"brain_parser"`.
      prefix for context parameters.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (features, num_epochs, gold_actions).
    features: A list of `feature_size` `Tensor` objects of type `string`. features firing at the current parser state, encoded as
      dist_belief.SparseFeatures protocol buffers.
    num_epochs: A `Tensor` of type `int32`. number of times this reader went over the training corpus.
    gold_actions: A `Tensor` of type `int32`. action to perform at the current parser state.
  """
  result = _op_def_lib.apply_op("GoldParseReader", task_context=task_context,
                                feature_size=feature_size,
                                batch_size=batch_size,
                                corpus_name=corpus_name,
                                arg_prefix=arg_prefix, name=name)
  return _GoldParseReaderOutput._make(result)


ops.RegisterShape("GoldParseReader")(None)
_lexicon_builder_outputs = [""]


def lexicon_builder(task_context, corpus_name=None,
                    lexicon_max_prefix_length=None,
                    lexicon_max_suffix_length=None, name=None):
  r"""An op that collects term statistics over a corpus and saves a set of term maps.

  Args:
    task_context: A `string`. file path at which to read the task context.
    corpus_name: An optional `string`. Defaults to `"documents"`.
      name of the context input to compute lexicons.
    lexicon_max_prefix_length: An optional `int`. Defaults to `3`.
      maximum prefix length for lexicon words.
    lexicon_max_suffix_length: An optional `int`. Defaults to `3`.
      maximum suffix length for lexicon words.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("LexiconBuilder", task_context=task_context,
                                corpus_name=corpus_name,
                                lexicon_max_prefix_length=lexicon_max_prefix_length,
                                lexicon_max_suffix_length=lexicon_max_suffix_length,
                                name=name)
  return result


ops.RegisterShape("LexiconBuilder")(None)
_projectivize_filter_outputs = ["filtered"]


def projectivize_filter(documents, task_context, corpus_name=None,
                        discard_non_projective=None, name=None):
  r"""TODO: add doc.

  Args:
    documents: A `Tensor` of type `string`.
    task_context: A `string`.
    corpus_name: An optional `string`. Defaults to `"documents"`.
    discard_non_projective: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  result = _op_def_lib.apply_op("ProjectivizeFilter", documents=documents,
                                task_context=task_context,
                                corpus_name=corpus_name,
                                discard_non_projective=discard_non_projective,
                                name=name)
  return result


ops.RegisterShape("ProjectivizeFilter")(None)
_unpack_sparse_features_outputs = ["indices", "ids", "weights"]


_UnpackSparseFeaturesOutput = collections.namedtuple("UnpackSparseFeatures",
                                                     _unpack_sparse_features_outputs)


def unpack_sparse_features(sf, name=None):
  r"""Converts a vector of strings with SparseFeatures to tensors.

  Note that indices, ids and weights are vectors of the same size and have
  one-to-one correspondence between their elements. ids and weights are each
  obtained by sequentially concatenating sf[i].id and sf[i].weight, for i in
  1...size(sf). Note that if sf[i].weight is not set, the default value for the
  weight is assumed to be 1.0. Also for any j, if ids[j] and weights[j] were
  extracted from sf[i], then index[j] is set to i.

  Args:
    sf: A `Tensor` of type `string`.
      vector of string, where each element is the string encoding of
      SpareFeatures proto.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (indices, ids, weights).
    indices: A `Tensor` of type `int32`. vector of indices inside sf
    ids: A `Tensor` of type `int64`. vector of id extracted from the SparseFeatures proto.
    weights: A `Tensor` of type `float32`. vector of weight extracted from the SparseFeatures proto.
  """
  result = _op_def_lib.apply_op("UnpackSparseFeatures", sf=sf, name=name)
  return _UnpackSparseFeaturesOutput._make(result)


ops.RegisterShape("UnpackSparseFeatures")(None)
_well_formed_filter_outputs = ["filtered"]


def well_formed_filter(documents, task_context, corpus_name=None,
                       keep_malformed_documents=None, name=None):
  r"""TODO: add doc.

  Args:
    documents: A `Tensor` of type `string`.
    task_context: A `string`.
    corpus_name: An optional `string`. Defaults to `"documents"`.
    keep_malformed_documents: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  result = _op_def_lib.apply_op("WellFormedFilter", documents=documents,
                                task_context=task_context,
                                corpus_name=corpus_name,
                                keep_malformed_documents=keep_malformed_documents,
                                name=name)
  return result


ops.RegisterShape("WellFormedFilter")(None)
_word_embedding_initializer_outputs = ["word_embeddings"]


def word_embedding_initializer(vectors, task_context, embedding_init=None,
                               name=None):
  r"""Reads word embeddings from an sstable of dist_belief.TokenEmbedding protos for

  every word specified in a text vocabulary file.

  Args:
    vectors: A `string`. path to recordio of word embedding vectors.
    task_context: A `string`. file path at which to read the task context.
    embedding_init: An optional `float`. Defaults to `1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
    a tensor containing word embeddings from the specified sstable.
  """
  result = _op_def_lib.apply_op("WordEmbeddingInitializer", vectors=vectors,
                                task_context=task_context,
                                embedding_init=embedding_init, name=name)
  return result


ops.RegisterShape("WordEmbeddingInitializer")(None)
def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "BeamEvalOutput"
  input_arg {
    name: "beam_state"
    type: DT_INT64
  }
  output_arg {
    name: "eval_metrics"
    type: DT_INT32
  }
  output_arg {
    name: "documents"
    type: DT_STRING
  }
  is_stateful: true
}
op {
  name: "BeamParseReader"
  output_arg {
    name: "features"
    type: DT_STRING
    number_attr: "feature_size"
  }
  output_arg {
    name: "beam_state"
    type: DT_INT64
  }
  output_arg {
    name: "num_epochs"
    type: DT_INT32
  }
  attr {
    name: "task_context"
    type: "string"
  }
  attr {
    name: "feature_size"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "beam_size"
    type: "int"
  }
  attr {
    name: "batch_size"
    type: "int"
    default_value {
      i: 1
    }
  }
  attr {
    name: "corpus_name"
    type: "string"
    default_value {
      s: "documents"
    }
  }
  attr {
    name: "allow_feature_weights"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "arg_prefix"
    type: "string"
    default_value {
      s: "brain_parser"
    }
  }
  attr {
    name: "continue_until_all_final"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "always_start_new_sentences"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "BeamParser"
  input_arg {
    name: "beam_state"
    type: DT_INT64
  }
  input_arg {
    name: "transition_scores"
    type: DT_FLOAT
  }
  output_arg {
    name: "features"
    type: DT_STRING
    number_attr: "feature_size"
  }
  output_arg {
    name: "next_beam_state"
    type: DT_INT64
  }
  output_arg {
    name: "alive"
    type: DT_BOOL
  }
  attr {
    name: "feature_size"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  is_stateful: true
}
op {
  name: "BeamParserOutput"
  input_arg {
    name: "beam_state"
    type: DT_INT64
  }
  output_arg {
    name: "indices_and_paths"
    type: DT_INT32
  }
  output_arg {
    name: "batches_and_slots"
    type: DT_INT32
  }
  output_arg {
    name: "gold_slot"
    type: DT_INT32
  }
  output_arg {
    name: "path_scores"
    type: DT_FLOAT
  }
  is_stateful: true
}
op {
  name: "DecodedParseReader"
  input_arg {
    name: "transition_scores"
    type: DT_FLOAT
  }
  output_arg {
    name: "features"
    type: DT_STRING
    number_attr: "feature_size"
  }
  output_arg {
    name: "num_epochs"
    type: DT_INT32
  }
  output_arg {
    name: "eval_metrics"
    type: DT_INT32
  }
  output_arg {
    name: "documents"
    type: DT_STRING
  }
  attr {
    name: "task_context"
    type: "string"
  }
  attr {
    name: "feature_size"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "batch_size"
    type: "int"
  }
  attr {
    name: "corpus_name"
    type: "string"
    default_value {
      s: "documents"
    }
  }
  attr {
    name: "arg_prefix"
    type: "string"
    default_value {
      s: "brain_parser"
    }
  }
  is_stateful: true
}
op {
  name: "DocumentSink"
  input_arg {
    name: "documents"
    type: DT_STRING
  }
  attr {
    name: "task_context"
    type: "string"
  }
  attr {
    name: "corpus_name"
    type: "string"
    default_value {
      s: "documents"
    }
  }
}
op {
  name: "DocumentSource"
  output_arg {
    name: "documents"
    type: DT_STRING
  }
  output_arg {
    name: "last"
    type: DT_BOOL
  }
  attr {
    name: "task_context"
    type: "string"
  }
  attr {
    name: "corpus_name"
    type: "string"
    default_value {
      s: "documents"
    }
  }
  attr {
    name: "batch_size"
    type: "int"
  }
  is_stateful: true
}
op {
  name: "FeatureSize"
  output_arg {
    name: "feature_sizes"
    type: DT_INT32
  }
  output_arg {
    name: "domain_sizes"
    type: DT_INT32
  }
  output_arg {
    name: "embedding_dims"
    type: DT_INT32
  }
  output_arg {
    name: "num_actions"
    type: DT_INT32
  }
  attr {
    name: "task_context"
    type: "string"
  }
  attr {
    name: "arg_prefix"
    type: "string"
    default_value {
      s: "brain_parser"
    }
  }
}
op {
  name: "GoldParseReader"
  output_arg {
    name: "features"
    type: DT_STRING
    number_attr: "feature_size"
  }
  output_arg {
    name: "num_epochs"
    type: DT_INT32
  }
  output_arg {
    name: "gold_actions"
    type: DT_INT32
  }
  attr {
    name: "task_context"
    type: "string"
  }
  attr {
    name: "feature_size"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "batch_size"
    type: "int"
  }
  attr {
    name: "corpus_name"
    type: "string"
    default_value {
      s: "documents"
    }
  }
  attr {
    name: "arg_prefix"
    type: "string"
    default_value {
      s: "brain_parser"
    }
  }
  is_stateful: true
}
op {
  name: "LexiconBuilder"
  attr {
    name: "task_context"
    type: "string"
  }
  attr {
    name: "corpus_name"
    type: "string"
    default_value {
      s: "documents"
    }
  }
  attr {
    name: "lexicon_max_prefix_length"
    type: "int"
    default_value {
      i: 3
    }
  }
  attr {
    name: "lexicon_max_suffix_length"
    type: "int"
    default_value {
      i: 3
    }
  }
}
op {
  name: "ProjectivizeFilter"
  input_arg {
    name: "documents"
    type: DT_STRING
  }
  output_arg {
    name: "filtered"
    type: DT_STRING
  }
  attr {
    name: "task_context"
    type: "string"
  }
  attr {
    name: "corpus_name"
    type: "string"
    default_value {
      s: "documents"
    }
  }
  attr {
    name: "discard_non_projective"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "UnpackSparseFeatures"
  input_arg {
    name: "sf"
    type: DT_STRING
  }
  output_arg {
    name: "indices"
    type: DT_INT32
  }
  output_arg {
    name: "ids"
    type: DT_INT64
  }
  output_arg {
    name: "weights"
    type: DT_FLOAT
  }
}
op {
  name: "WellFormedFilter"
  input_arg {
    name: "documents"
    type: DT_STRING
  }
  output_arg {
    name: "filtered"
    type: DT_STRING
  }
  attr {
    name: "task_context"
    type: "string"
  }
  attr {
    name: "corpus_name"
    type: "string"
    default_value {
      s: "documents"
    }
  }
  attr {
    name: "keep_malformed_documents"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "WordEmbeddingInitializer"
  output_arg {
    name: "word_embeddings"
    type: DT_FLOAT
  }
  attr {
    name: "vectors"
    type: "string"
  }
  attr {
    name: "task_context"
    type: "string"
  }
  attr {
    name: "embedding_init"
    type: "float"
    default_value {
      f: 1
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
