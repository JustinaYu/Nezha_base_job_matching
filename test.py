from transformers import AutoTokenizer

# # 加载 tokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#
# # 查看原始词汇
# print("Original vocab size:", len(tokenizer))
#
# # 定义要替换的 token
# old_token = "old_token"
# new_token = "new_token"
#
# # 添加新 token
# tokenizer.add_tokens([new_token])
#
# # 删除旧 token
# if old_token in tokenizer.get_vocab():
#     old_token_id = tokenizer.get_vocab()[old_token]
#     # 将旧 token 的 id 设置为 tokenizer.unk_token_id
#     tokenizer.vocab[old_token] = tokenizer.unk_token_id
#
# # 检查词汇表大小
# print("Updated vocab size:", len(tokenizer))

# n-gram

def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  # Note(mingdachen): We create a list for recording if the piece is
  # the starting piece of current token, where 1 means true, so that
  # on-the-fly whole word masking is possible.
  token_boundary = [0] * len(tokens)

  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      token_boundary[i] = 1
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.
    if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
        not is_start_piece(token)):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])
      if is_start_piece(token):
        token_boundary[i] = 1

  output_tokens = list(tokens)

  masked_lm_positions = []
  masked_lm_labels = []

  if masked_lm_prob == 0:
    return (output_tokens, masked_lm_positions,
            masked_lm_labels, token_boundary)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  # Note(mingdachen):
  # By default, we set the probilities to favor shorter ngram sequences.
  ngrams = np.arange(1, FLAGS.ngram + 1, dtype=np.int64)
  pvals = 1. / np.arange(1, FLAGS.ngram + 1)
  pvals /= pvals.sum(keepdims=True)

  if not FLAGS.favor_shorter_ngram:
    pvals = pvals[::-1]

  ngram_indexes = []
  for idx in range(len(cand_indexes)):
    ngram_index = []
    for n in ngrams:
      ngram_index.append(cand_indexes[idx:idx+n])
    ngram_indexes.append(ngram_index)

  rng.shuffle(ngram_indexes)

  masked_lms = []
  covered_indexes = set()
  for cand_index_set in ngram_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    if not cand_index_set:
      continue
    # Note(mingdachen):
    # Skip current piece if they are covered in lm masking or previous ngrams.
    for index_set in cand_index_set[0]:
      for index in index_set:
        if index in covered_indexes:
          continue

    n = np.random.choice(ngrams[:len(cand_index_set)],
                         p=pvals[:len(cand_index_set)] /
                         pvals[:len(cand_index_set)].sum(keepdims=True))
    index_set = sum(cand_index_set[n - 1], [])
    n -= 1
    # Note(mingdachen):
    # Repeatedly looking for a candidate that does not exceed the
    # maximum number of predictions by trying shorter ngrams.
    while len(masked_lms) + len(index_set) > num_to_predict:
      if n == 0:
        break
      index_set = sum(cand_index_set[n - 1], [])
      n -= 1
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8:
        masked_token = "[MASK]"
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5:
          masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

      output_tokens[index] = masked_token

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  assert len(masked_lms) <= num_to_predict

  rng.shuffle(ngram_indexes)

  select_indexes = set()

  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)
  return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary)
