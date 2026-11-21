import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # TensorFlow 안내 로그 줄이기

import json

import numpy
import pandas

# 학습 결과물(단어 벡터, 단어 사전, 불용어)을 두는 폴더 (환경 변수 MODEL_DIR로 바꿀 수 있음)
MODEL_DIR = os.getenv('MODEL_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model'))

# 단어 사전 생성
# min_count 미만으로 등장한 드문 단어는 학습할 문맥이 부족하므로 사전에서 제외
# 반환: (IDS 열이 추가된 df, word2id, id2word)
#   - word2id: 단어 → 정수 ID (많이 나온 단어일수록 작은 번호, 0은 패딩/모르는 단어용으로 비워 둠)
#   - id2word: 정수 ID → 단어
#   - IDS 열: NOUNS의 단어들을 정수 ID로 바꾼 리스트 (사전에 없는 단어는 빠짐)
def build_vocab(df: pandas.DataFrame, min_count: int = 3) -> tuple[pandas.DataFrame, dict[str, int], dict[int, str]]:
    counts = df['NOUNS'].explode().value_counts()
    vocab = counts[counts >= min_count].index

    word2id = {word: i + 1 for i, word in enumerate(vocab)}
    id2word = {i: word for word, i in word2id.items()}

    df = df.assign(IDS=df['NOUNS'].apply(lambda nouns: [word2id[w] for w in nouns if w in word2id]))
    return df, word2id, id2word


# skip-gram 학습 쌍 생성
# 각 단어(target) 주변 window 안의 단어를 정답 문맥으로 하고, 오답(negative) num_ns개를 붙인다.
# 오답은 단어 빈도의 0.75제곱 비율로 뽑는다 (자주 나오는 단어가 조금 더 자주 뽑힘).
# 반환: targets (N,), contexts (N, 1+num_ns), labels (N, 1+num_ns) — contexts/labels의 0번째가 정답
def make_pairs(sentences: pandas.Series, vocab_size: int, window: int, num_ns: int,
               rng: numpy.random.Generator) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    targets, positives = [], []
    for ids in sentences:
        for i, target in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if i != j:
                    targets.append(target)
                    positives.append(ids[j])

    counts = numpy.bincount(sentences.explode().dropna().astype(int), minlength=vocab_size).astype(float)
    prob = counts ** 0.75
    prob[0] = 0  # 0번(패딩)은 뽑지 않음
    prob /= prob.sum()

    targets = numpy.array(targets, dtype='int32')
    negatives = rng.choice(vocab_size, size=(len(targets), num_ns), p=prob).astype('int32')
    contexts = numpy.concatenate([numpy.array(positives, dtype='int32')[:, None], negatives], axis=1)

    labels = numpy.zeros(contexts.shape, dtype='float32')
    labels[:, 0] = 1
    return targets, contexts, labels


# Keras로 Word2Vec 학습 → 단어 벡터 행렬 (vocab_size, dim), 행마다 길이 1로 정규화
def train(sentences: pandas.Series, vocab_size: int, dim: int = 100, window: int = 2, num_ns: int = 4,
          epochs: int = 10, batch_size: int = 1024, seed: int = 42) -> numpy.ndarray:
    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(seed)  # 매번 같은 결과가 나오도록 고정
    targets, contexts, labels = make_pairs(sentences, vocab_size, window, num_ns, numpy.random.default_rng(seed))

    class Word2Vec(keras.Model):
        def __init__(self):
            super().__init__()
            self.target_embedding = keras.layers.Embedding(vocab_size, dim, name='target')
            self.context_embedding = keras.layers.Embedding(vocab_size, dim, name='context')

        def call(self, inputs):
            target, context = inputs
            t = self.target_embedding(target)            # (batch, dim)
            c = self.context_embedding(context)          # (batch, 1+num_ns, dim)
            return tf.einsum('bd,bnd->bn', t, c)         # 후보마다 점수 (내적)

    model = Word2Vec()
    model.compile(optimizer='adam',
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit((targets, contexts), labels, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

    vectors = model.target_embedding.get_weights()[0]
    return normalize(vectors)


def normalize(vectors: numpy.ndarray) -> numpy.ndarray:
    norm = numpy.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / numpy.maximum(norm, 1e-9)


# 학습된 단어 벡터로 유사 단어 찾기, 법률안/검색어를 벡터로 바꾸기
class WordVectors:
    def __init__(self, vectors: numpy.ndarray, word2id: dict[str, int], id2word: dict[int, str]):
        self._vectors = vectors
        self._word2id = word2id
        self._id2word = id2word
        self._doc_mean = numpy.zeros(vectors.shape[1])

    # 단어와 가장 비슷한 단어 topn개 [(단어, 코사인 유사도), ...]  (사전에 없는 단어면 빈 리스트)
    def most_similar(self, word: str, topn: int = 5) -> list[tuple[str, float]]:
        if word not in self._word2id:
            return []
        sims = self._vectors @ self._vectors[self._word2id[word]]
        sims[0] = -numpy.inf                        # 패딩 제외
        sims[self._word2id[word]] = -numpy.inf      # 자기 자신 제외
        best = numpy.argsort(-sims)[:topn]
        return [(self._id2word[i], round(float(sims[i]), 4)) for i in best]

    # 단어 ID 리스트(또는 numpy 배열)의 평균 벡터 (단어가 없으면 None)
    def _mean(self, ids) -> numpy.ndarray | None:
        return self._vectors[ids].mean(axis=0) if len(ids) else None

    # 법률안마다 명사 벡터의 평균 → (법안 수, dim)
    # 모든 법안에 공통으로 섞인 성분(전체 평균)을 빼야 법안끼리 구분이 잘 된다.
    def fit_documents(self, ids_series: pandas.Series) -> numpy.ndarray:
        dim = self._vectors.shape[1]
        docs = numpy.array([self._mean(ids) if len(ids) else numpy.zeros(dim) for ids in ids_series],
                           dtype='float32')
        self._doc_mean = docs.mean(axis=0)
        return normalize(docs - self._doc_mean)

    # 검색어 단어들 → 벡터 (사전에 있는 단어가 하나도 없으면 None)
    def query_vector(self, words: list[str]) -> numpy.ndarray | None:
        mean = self._mean([self._word2id[w] for w in words if w in self._word2id])
        return None if mean is None else normalize(mean - self._doc_mean)


# 학습 결과물: 단어 벡터 + 단어 사전 + 불용어
# - 로컬에서 train.py로 만들어 저장하고, 서버는 불러오기만 한다 (서버에는 TensorFlow가 필요 없음)
# - 파일 이름과 형식은 이 클래스 안에서만 다룬다
class Model:
    _VECTORS = 'vectors.npy'
    _VOCAB = 'vocab.json'

    def __init__(self, vectors: numpy.ndarray, word2id: dict[str, int], stopwords: list[str]):
        self._vectors = vectors.astype('float32')
        self._word2id = dict(word2id)
        self._id2word = {i: w for w, i in self._word2id.items()}
        self._stopwords = frozenset(stopwords)

    @property
    def stopwords(self) -> frozenset[str]:
        return self._stopwords

    @property
    def word2id(self) -> dict[str, int]:
        return dict(self._word2id)

    @property
    def id2word(self) -> dict[int, str]:
        return dict(self._id2word)

    # 단어 리스트 → 단어 ID 배열 (int32, 사전에 없는 단어는 빠짐)
    def to_ids(self, words: list[str]) -> numpy.ndarray:
        return numpy.array([self._word2id[w] for w in words if w in self._word2id], dtype='int32')

    def word_vectors(self) -> WordVectors:
        return WordVectors(self._vectors, self._word2id, self._id2word)

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        words = [''] * (len(self._word2id) + 1)  # 0번은 패딩
        for word, i in self._word2id.items():
            words[i] = word
        numpy.save(os.path.join(directory, self._VECTORS), self._vectors)
        with open(os.path.join(directory, self._VOCAB), 'w', encoding='utf-8') as f:
            json.dump({'words': words, 'stopwords': sorted(self._stopwords)}, f, ensure_ascii=False)

    # 저장된 모델 불러오기 (없거나 깨졌으면 None)
    @classmethod
    def load(cls, directory: str) -> 'Model | None':
        try:
            vectors = numpy.load(os.path.join(directory, cls._VECTORS))
            with open(os.path.join(directory, cls._VOCAB), encoding='utf-8') as f:
                vocab = json.load(f)
            words = vocab['words']
            if len(words) != len(vectors):
                return None
            return cls(vectors, {w: i for i, w in enumerate(words) if i > 0}, vocab['stopwords'])
        except (OSError, ValueError, KeyError, TypeError):
            return None
