from direct_redis import DirectRedis

from langframe import *
from train.functions import *
from transformer.Models import *

REDIS = True

r = DirectRedis(host='127.0.0.1', port='6379')
device = torch.device('cuda:0')
tokenizer_kor = KoreanTokenizer('mecab')


def get_corpus_preprocessed():
    df = pd.read_excel('data/korean-english-parallel/2_대화체_200226.xlsx')
    df = df[df['대분류'] == '일상대화']
    kor, eng = df['원문'].tolist(), df['번역문'].tolist()

    tokenized_matrix_eng = get_tokenized_matrix(eng, 'word_tokenize', False, [])
    tokenized_matrix_kor = [tokenizer_kor.morphs(text) for text in kor]

    # add <sos>, <eos> tokens
    tokenized_matrix_eng = [["<sos>"] + tokens + ["<eos>"] for tokens in tokenized_matrix_eng]
    tokenized_matrix_kor = [["<sos>"] + tokens + ["<eos>"] for tokens in tokenized_matrix_kor]

    # token2idx, idx2token 생성
    unique_tokens_eng, unique_tokens_kor = get_uniques_from_nested_lists(
        tokenized_matrix_eng), get_uniques_from_nested_lists(tokenized_matrix_kor)
    token2idx_eng, idx2token_eng = get_item2idx(unique_tokens_eng, unique=True, start_from_one=True)
    token2idx_kor, idx2token_kor = get_item2idx(unique_tokens_kor, unique=True, start_from_one=True)

    # max_seq_len
    max_seq_len_enc = get_max_seq_len(tokenized_matrix_kor)
    max_seq_len_dec = get_max_seq_len(tokenized_matrix_eng)

    # making sequence-wise inputs
    X_enc, X_dec, y_true = [], [], []
    for idx, tokens in enumerate(tokenized_matrix_eng):
        X_enc.extend([tokenized_matrix_kor[idx] for _ in range(len(tokens) - 1)])
        X_dec.extend([tokens[:seq_i] for seq_i in range(1, len(tokens))])
        y_true.extend(tokens[1:])

    # padding
    padded_enc = pad_sequence_nested_lists(X_enc, max_len=max_seq_len_enc, method='post', truncating='post')
    padded_dec = pad_sequence_nested_lists(X_dec, max_len=max_seq_len_dec, method='post', truncating='post')
    X_enc = [[token2idx_kor[token] for token in row] for row in padded_enc]
    X_dec = [[token2idx_eng[token] for token in row] for row in padded_dec]
    y_true = [token2idx_eng[token] for token in y_true]
    corpus_set = [X_enc, X_dec, y_true, unique_tokens_kor, unique_tokens_eng, max_seq_len_enc, max_seq_len_dec]

    res = r.hset('keparallel', 'ilsang', corpus_set)
    print(f"Set redis reponse: {res}")
    return corpus_set
