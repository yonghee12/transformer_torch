from itertools import chain

from random import choice
from progress_timer import Timer
from direct_redis import DirectRedis

from transformer.Models import *

r = DirectRedis(host='127.0.0.1', port='6379')
model_name = 'glove.6B.300d'

get_from_redis = True
if get_from_redis:
    corpus_set = r.hget('bible_corpus', f'{model_name}_1k')
else:
    raise FileNotFoundError

X_vectors, total_used_tokens, y_true, y_true_stack, unique_tokens, \
total_used_tokens, idx2token, token2idx = corpus_set
flattened_used_tokens = list(set(chain.from_iterable(total_used_tokens)))

print(X_vectors.shape)
print(y_true.shape)
print(y_true_stack.shape)
print(len(unique_tokens))
whole_batch = X_vectors.shape[0]

model = Transformer()


# model = RNNTrainer(input_dim=X_vectors.shape[2], hidden_dim=1000, output_size=len(unique_tokens),
#                    layer='lstm', backend='numpy', timemethod='stack', stateful=False)
# model.fit(X_vectors, y_true_stack, batch_size=whole_batch // 20, lr=0.3, n_epochs=500, print_many=False, verbose=1)
# print()
#
# max_timesteps = 4
# for start_word in flattened_used_tokens[100:300]:
#     generated = [start_word]
#     for t in range(max_timesteps):
#         try:
#             word_vectors = np.array([r.hget("glove_2.2m_300d", word) for word in generated])
#             predicted_idx = model.predict(word_vectors)
#             generated.append(idx2token[predicted_idx])
#         except Exception as e:
#             print(str(e))
#     print(generated)
#
#
# print()
# # Generation logic
# # test_sequences = get_sequences_from_tokens_window(unique_tokens, token2idx, window_size=2)
# timer = Timer(len(total_used_tokens))
# gens = []
# for idx, test_sequence in enumerate(total_used_tokens):
#     timer.time_check(idx)
#     x_test = ' '.join(test_sequence[:-1])
#     generated = predict_rnn_lm_bible_windowed(x_test, wv, idx2token, indexed=False)
#     output_str = x_test + ' ' + generated
#     gens.append(output_str)
#     # print(output_str)

# Random generation logic
# random_gens = []
# for _ in range(1000):
#     w1, w2 = choice(unique_tokens), choice(unique_tokens)
#     w3 = predict_rnn_lm_bible_windowed(' '.join([w1, w2]), wv, idx2token, indexed=False)
#     w4 = predict_rnn_lm_bible_windowed(' '.join([w2, w3]), wv, idx2token, indexed=False)
#     output_str = ' '.join([w1, w2, w3, w4])
#     random_gens.append(output_str)
#
# loss_val = '0.015'
# with open(f'results/bible_generated_w3_loss{loss_val}.txt', 'w') as f:
#     f.write('\n'.join(gens))
#     f.close()
#
# with open(f'results/bible_random_generated_w3_loss{loss_val}.txt', 'w') as f:
#     f.write('\n'.join(random_gens))
#     f.close()
#
# orgns = [' '.join(tokens) for tokens in total_used_tokens]
# with open(f'results/bible_original_w3_loss{loss_val}.txt', 'w') as f:
#     f.write('\n'.join(orgns))
#     f.close()
