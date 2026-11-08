import pandas

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
