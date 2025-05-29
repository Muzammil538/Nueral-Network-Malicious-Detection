def char_tokenizer(url, max_len=200):
    char_set = "abcdefghijklmnopqrstuvwxyz0123456789-._:/@"
    char2idx = {c: i + 1 for i, c in enumerate(char_set)}
    tokens = [char2idx.get(c, 0) for c in url.lower()]
    return tokens[:max_len] + [0] * max(0, max_len - len(tokens))
