# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Utility functions to parse and format the raw HOL statement text files.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import random
import numpy as np


class DataParser(object):

  def __init__(self, source_dir, use_tokens=False, verbose=1):
    random.seed(1337)
    self.use_tokens = use_tokens
    if self.use_tokens:
      self.step_markers = {'T'}

      def tokenize_fn(line):
        line = line.rstrip()[2:]
        tokens = line.split()
        return tokens
      self.tokenize_fn = tokenize_fn
    else:
      self.step_markers = {'+', '-', 'C', 'A'}
      self.tokenize_fn = lambda x: x.rstrip()[2:]
    self.verbose = verbose
    train_dir = os.path.join(source_dir, 'train')
    val_dir = os.path.join(source_dir, 'test')
    self.train_fnames = [
        os.path.join(train_dir, '%05d' % i) for i in range(1, 10000)]
    self.val_fnames = [
        os.path.join(val_dir, fname)
        for fname in os.listdir(val_dir) if fname.isdigit()]
    if verbose:
      logging.info('Building vocabulary...')
    self.vocabulary = self.build_vocabulary()
    if verbose:
      logging.info('Found %s unique tokens.', len(self.vocabulary))
    self.vocabulary_index = dict(enumerate(self.vocabulary))
    self.reverse_vocabulary_index = dict(
        [(value, key) for (key, value) in self.vocabulary_index.items()])
    self.train_conjectures = self.parse_file_list(self.train_fnames)
    self.val_conjectures = self.parse_file_list(self.val_fnames)
    self.train_conjectures_names = sorted(self.train_conjectures.keys())
    self.val_conjectures_names = sorted(self.val_conjectures.keys())

  def build_vocabulary(self):
    vocabulary = set()
    for fname in self.train_fnames:
      f = open(fname)
      for line in f:
        if line[0] in self.step_markers:
          for token in self.tokenize_fn(line):
            vocabulary.add(token)
      f.close()
    return vocabulary

  def parse_file_list(self, fnames):
    conjectures = {}
    for fname in fnames:
      conjecture = self.parse_file(fname)
      name = conjecture.pop('name')
      conjectures[name] = conjecture
    return conjectures

  def display_stats(self, conjectures):
    dep_counts = []
    dep_lengths = []
    conj_lengths = []
    pos_step_counts = []
    pos_step_lengths = []
    neg_step_counts = []
    neg_step_lengths = []

    logging.info('%s conjectures in total.', len(conjectures))
    for value in conjectures.values():
      deps = value['deps']
      conj = value['conj']
      pos_steps = value['+']
      neg_steps = value['-']
      dep_counts.append(len(deps))
      if deps:
        dep_lengths.append(np.mean([len(x) for x in deps]))
      conj_lengths.append(len(conj))
      pos_step_counts.append(len(pos_steps))
      if pos_steps:
        pos_step_lengths.append(np.mean([len(x) for x in pos_steps]))
      neg_step_counts.append(len(neg_steps))
      if neg_steps:
        neg_step_lengths.append(np.mean([len(x) for x in neg_steps]))
    logging.info('total number of steps: %s',
                 np.sum(pos_step_counts) + np.sum(neg_step_counts))
    logging.info('mean number of positive steps per conjecture: %s',
                 np.mean(pos_step_counts))
    logging.info('mean number of negative steps per conjecture: %s',
                 np.mean(neg_step_counts))
    logging.info('mean conjecture length: %s', np.mean(conj_lengths))
    logging.info('mean number of dependencies: %s', np.mean(dep_counts))
    logging.info('mean dependency length: %s', np.mean(dep_lengths))
    logging.info('mean number of positive steps: %s',
                 np.mean(pos_step_counts))
    logging.info('mean number of negative steps: %s',
                 np.mean(neg_step_counts))
    logging.info('mean positive step length: %s', np.mean(pos_step_lengths))
    logging.info('mean negative step length: %s', np.mean(neg_step_lengths))
    # TODO(fchollet): plot histograms

  def parse_file(self, fname):
    f = open(fname)
    name = f.readline().rstrip()[2:]
    if self.use_tokens:
      # Text representation of conjecture.
      f.readline()
      # Tokenization of conjecture.
      conj = self.tokenize_fn(f.readline())
    else:
      # Text representation of conjecture.
      conj = self.tokenize_fn(f.readline())
    conjecture = {
        'name': name,
        'deps': [],
        '+': [],
        '-': [],
        'conj': conj,
    }
    while 1:
      line = f.readline()
      if not line:
        break
      marker = line[0]
      if self.use_tokens:
        line = f.readline()  # Text representation
      content = self.tokenize_fn(line)
      if marker == 'D':
        conjecture['deps'].append(content)
      elif marker == '+':
        conjecture['+'].append(content)
      elif marker == '-':
        conjecture['-'].append(content)
    return conjecture

  def integer_encode_statements(self, statements, max_len):
    encoded = np.zeros((len(statements), max_len), dtype='int32')
    for s, statement in enumerate(statements):
      for i, char in enumerate(statement[:max_len]):
        encoded[s, max_len - i - 1] = self.reverse_vocabulary_index.get(
            char, -1) + 1
    return encoded

  def one_hot_encode_statments(self, statements, max_len):
    encoded = np.zeros((len(statements), max_len, len(self.vocabulary) + 1),
                       dtype='float32')
    for s, statement in enumerate(statements):
      for i, char in enumerate(statement[:max_len]):
        j = self.reverse_vocabulary_index.get(char, -1) + 1
        encoded[s, max_len - i - 1, j] = 1
    return encoded

  def draw_random_batch_of_steps(self, split='train', encoding='integer',
                                 max_len=256, batch_size=128):
    if split == 'train':
      all_conjectures = self.train_conjectures
    elif split == 'val':
      all_conjectures = self.val_conjectures
    else:
      raise ValueError('`split` must be in {"train", "val"}.')
    if encoding == 'integer':
      encode = lambda x: self.integer_encode_statements(x, max_len=max_len)
    elif encoding == 'one-hot':
      encode = lambda x: self.one_hot_encode_statments(x, max_len=max_len)
    else:
      raise ValueError('Unknown encoding:', encoding)

    labels = np.random.randint(0, 2, size=(batch_size,))
    steps = []
    i = 0
    while len(steps) < batch_size:
      name = random.choice(all_conjectures.keys())
      conjecture = all_conjectures[name]
      if labels[i]:
        conjecture_steps = conjecture['+']
      else:
        conjecture_steps = conjecture['-']
      if not conjecture_steps:
        continue
      step = random.choice(conjecture_steps)
      steps.append(step)
      i += 1
    return encode(steps), labels

  def draw_batch_of_steps_in_order(self, conjecture_index=0, step_index=0,
                                   split='train',
                                   encoding='integer',
                                   max_len=256, batch_size=128):
    if split == 'train':
      all_conjectures = self.train_conjectures
      conjecture_names = self.train_conjectures_names
    elif split == 'val':
      all_conjectures = self.val_conjectures
      conjecture_names = self.val_conjectures_names
    else:
      raise ValueError('`split` must be in {"train", "val"}.')
    if encoding == 'integer':
      encode = lambda x: self.integer_encode_statements(x, max_len=max_len)
    elif encoding == 'one-hot':
      encode = lambda x: self.one_hot_encode_statments(x, max_len=max_len)
    else:
      raise ValueError('Unknown encoding:', encoding)
    labels = []
    steps = []
    while len(steps) < batch_size:
      conj_name = conjecture_names[conjecture_index % len(conjecture_names)]
      conjecture = all_conjectures[conj_name]
      conjecture_steps = conjecture['+'] + conjecture['-']
      if len(conjecture_steps) > step_index:
        step_labels = ([1] * len(conjecture['+']) +
                       [0] * len(conjecture['-']))
        remaining = batch_size - len(steps)
        steps += conjecture_steps[step_index: step_index + remaining]
        labels += step_labels[step_index: step_index + remaining]
        step_index += remaining
      else:
        step_index = 0
        conjecture_index += 1
    labels = np.asarray(labels).astype('float32')
    return (encode(steps), labels), (conjecture_index, step_index)

  def draw_batch_of_steps_and_conjectures_in_order(
      self, conjecture_index=0, step_index=0,
      split='train', encoding='integer', max_len=256, batch_size=128):
    if split == 'train':
      all_conjectures = self.train_conjectures
      conjecture_names = self.train_conjectures_names
    elif split == 'val':
      all_conjectures = self.val_conjectures
      conjecture_names = self.val_conjectures_names
    else:
      raise ValueError('`split` must be in {"train", "val"}.')
    if encoding == 'integer':
      encode = lambda x: self.integer_encode_statements(x, max_len=max_len)
    elif encoding == 'one-hot':
      encode = lambda x: self.one_hot_encode_statments(x, max_len=max_len)
    else:
      raise ValueError('Unknown encoding:', encoding)
    labels = []
    conjectures = []
    steps = []
    while len(steps) < batch_size:
      conj_name = conjecture_names[conjecture_index % len(conjecture_names)]
      conjecture = all_conjectures[conj_name]
      conjecture_steps = conjecture['+'] + conjecture['-']
      if len(conjecture_steps) > step_index:
        step_labels = ([1] * len(conjecture['+']) +
                       [0] * len(conjecture['-']))
        remaining = batch_size - len(steps)
        steps += conjecture_steps[step_index: step_index + remaining]
        added_labels = step_labels[step_index: step_index + remaining]
        labels += added_labels
        conjectures += [conjecture['conj']] * len(added_labels)
        step_index += remaining
      else:
        step_index = 0
        conjecture_index += 1
    labels = np.asarray(labels).astype('float32')
    return (([encode(steps), encode(conjectures)], labels),
            (conjecture_index, step_index))

  def draw_random_batch_of_steps_and_conjectures(self, split='train',
                                                 encoding='integer',
                                                 max_len=256,
                                                 batch_size=128):
    if split == 'train':
      all_conjectures = self.train_conjectures
    elif split == 'val':
      all_conjectures = self.val_conjectures
    else:
      raise ValueError('`split` must be in {"train", "val"}.')
    if encoding == 'integer':
      encode = lambda x: self.integer_encode_statements(x, max_len=max_len)
    elif encoding == 'one-hot':
      encode = lambda x: self.one_hot_encode_statments(x, max_len=max_len)
    else:
      raise ValueError('Unknown encoding:', encoding)

    labels = np.random.randint(0, 2, size=(batch_size,))
    conjectures = []
    steps = []
    i = 0
    while len(steps) < batch_size:
      name = random.choice(all_conjectures.keys())
      conjecture = all_conjectures[name]
      if labels[i]:
        conjecture_steps = conjecture['+']
      else:
        conjecture_steps = conjecture['-']
      if not conjecture_steps:
        continue
      step = random.choice(conjecture_steps)
      conjectures.append(conjecture['conj'])
      steps.append(step)
      i += 1
    return [encode(steps), encode(conjectures)], labels

  def training_steps_generator(self, encoding='integer',
                               max_len=256, batch_size=128):
    while 1:
      yield self.draw_random_batch_of_steps(
          'train', encoding, max_len, batch_size)

  def validation_steps_generator(self, encoding='integer',
                                 max_len=256, batch_size=128):
    conj_index, step_index = 0, 0
    while 1:
      data, (conj_index, step_index) = self.draw_batch_of_steps_in_order(
          conj_index, step_index, 'val', encoding, max_len, batch_size)
      yield data

  def training_steps_and_conjectures_generator(
      self, encoding='integer', max_len=256, batch_size=128):
    while 1:
      yield self.draw_random_batch_of_steps_and_conjectures(
          'train', encoding, max_len, batch_size)

  def validation_steps_and_conjectures_generator(
      self, encoding='integer', max_len=256, batch_size=128):
    conj_index, step_index = 0, 0
    while 1:
      fn = self.draw_batch_of_steps_and_conjectures_in_order
      data, (conj_index, step_index) = fn(
          conj_index, step_index, 'val', encoding, max_len, batch_size)
      yield data
