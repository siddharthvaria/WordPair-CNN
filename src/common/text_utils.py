#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import os
import ftfy
import json
from collections import defaultdict
import spacy
# import tldextract
from tqdm import tqdm
import math
import string

# puncs = u'#$%&\()*+-/<=>@[\\]^_`{|}~'
puncs_removed = u'!"#$%&\'()*+-/<=>@[\\]^_`{|}~'
puncs_retained = u',.:;?'

# def get_domain_name(url_text):
#     e = tldextract.extract(url_text)
#     return u'.'.join(part for part in e[1:] if part)


def remove_repeated_chars(text):
    return re.sub(r'(.)\1+', r'\1\1', text)


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

# def text_standardize(text):
#     """
#     fixes some issues the spacy tokenizer had on books corpus
#     also does some whitespace standardization
#     """
#     text = text.replace('—', '-')
#     text = text.replace('–', '-')
#     text = text.replace('―', '-')
#     text = text.replace('…', '...')
#     text = text.replace('´', "'")
#     text = re.sub(r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
#     text = re.sub(r'\s*\n\s*', ' \n ', text)
#     text = re.sub(r'[^\S\n]+', ' ', text)
#     return text.strip()


def clean(text):
    text = text.strip()
    text = re.sub(r"\(.+?\)", r"", text)  # remove text within brackets. That is strings like "(blah blah)"
    text = re.sub(r'\s+', r' ', text)
    return text.strip()


class SubwordEncoder(object):
    """
    subword encoder (uses bpe encoding)
    """

    def __init__(self, encoder_path, bpe_path, to_file = None):
        self.nlp = spacy.load('en_core_web_md', disable = ['parser', 'tagger', 'ner', 'textcat'])
        self.encoder = json.load(open(encoder_path))
        self.decoder = {v:k for k, v in self.encoder.items()}
        merges = open(bpe_path, encoding = 'utf-8').read().split('\n')[1:-1]  # leave the first split and last split
        merges = [tuple(merge.split()) for merge in merges]
        # At this point merges is a list of tuples
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        # bpe_ranks is a dict from tuples to integers
        # tuples are of the form: (u'wo', u'ol'), (u'mari', u'onette</w>'), (u'amphi', u'theater</w>') etc
        self.cache = {}
        self.to_file = to_file
        if self.to_file:
            # remove the file if it exists
            try:
                os.remove(self.to_file)
            except OSError:
                pass

    def bpe(self, token):
        # token is single lower case word
        word = tuple(token[:-1]) + (token[-1] + u'</w>',)
        # if token is 'good' then word will be ('g', 'o', 'o', 'd</w>')
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token + u'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        if word == u'\n  </w>':
            word = u'\n</w>'
        self.cache[token] = word
        return word

    def encode(self, texts):
        texts_tokens = []
        for text in tqdm(texts, ncols = 80, leave = False):
            # For ftfy.fix_text see https://ftfy.readthedocs.io/en/latest/
            text = self.nlp(ftfy.fix_text(text))
            # At this point text is of type spacy.tokens.doc.Doc
            text_tokens = []
            for token in text:
                text_tokens.extend([self.encoder.get(t, 0) for t in self.bpe(token.text.lower()).split(' ')])
            texts_tokens.append(text_tokens)

        if self.to_file:
            with open(self.to_file, 'a', encoding = 'utf8') as fhw:
                for text_tokens in texts_tokens:
                    text = []
                    for text_id in text_tokens:
                        text_token = self.decoder[text_id]
                        if text_token.endswith(u'</w>'):
                            text_token = text_token[:-len(u'</w>')]
                        text.append(text_token)
                    text = u' '.join(text)
                    fhw.write(text)
                    fhw.write(u'\n-----------------\n')

        return texts_tokens


class WordEncoder(object):
    """
    word encoder
    """

    def __init__(self, to_file = None):
        self.nlp = spacy.load('en_core_web_md', disable = ['parser', 'textcat'])

        self.encoder = defaultdict(int)  # dict from word to index
        self.encoder[u'<PAD>'] = 0

        self.decoder = {}
        self.decoder[0] = u'<PAD>'

        self.encoder_pos = defaultdict(int)
        self.encoder_pos[u'<PAD>'] = 0

        self.decoder_pos = {}
        self.decoder_pos[0] = u'<PAD>'

        self.encoder_ner = defaultdict(int)
        self.encoder_ner[u'<PAD>'] = 0

        self.decoder_ner = {}
        self.decoder_ner[0] = u'<PAD>'

        self.to_file = to_file
        if self.to_file:
            # remove the file if it exists
            try:
                os.remove(self.to_file)
            except OSError:
                pass

        self.wplen_dist = defaultdict(int)

    def word2index(self, word):
        if word in self.encoder:
            return self.encoder[word]
        _len = len(self.encoder)
        self.encoder[word] = _len
        self.decoder[_len] = word
        return _len

    def pos2index(self, tag):
        if tag in self.encoder_pos:
            return self.encoder_pos[tag]
        _len = len(self.encoder_pos)
        self.encoder_pos[tag] = _len
        self.decoder_pos[_len] = tag
        return _len

    def ner2index(self, ner):
        if ner in self.encoder_ner:
            return self.encoder_ner[ner]
        _len = len(self.encoder_ner)
        self.encoder_ner[ner] = _len
        self.decoder_ner[_len] = ner
        return _len

    def _encode_args(self, text, max_arg_len = 100):
        # .text : text
        # .ent_iob_, .ent_type_ : ner tag
        # .tag_ : pos_tag
        text_tokens = []
        pos_tags = []
        ner_tags = []
        _len = 0

        for ii, token in enumerate(text):
            # only retain most common english punctuation. That is ',.:;?'
            if len(token.text) == 1 and token.text in puncs_removed:
                continue
            pos_tags.append(self.pos2index(token.tag_))
            ner_tags.append(self.ner2index((token.ent_iob_ + '_' + token.ent_type_) if len(token.ent_type_) > 0 else token.ent_iob_))
            if token.like_num:
                text_tokens.append(self.word2index(u"NUM"))
            else:
                text_tokens.append(self.word2index(token.text))
            _len += 1
            if _len >= max_arg_len:
                break

        return text_tokens, pos_tags, ner_tags

    def _encode_wp(self, text):
        # .text : text
        # .ent_iob_, .ent_type_ : ner tag
        # .tag_ : pos_tag
        text_tokens = []
        pos_tags = []
        ner_tags = []

        for token in text:

            if len(token.text) < 3 and (token.text[0] not in string.ascii_uppercase) and (token.text not in ['as', 'or', 'so', 'if', 'no', 'do', 'he', 'we', 'me', 'us', 'it']):
                continue
            # if word contains anything other than A-Za-z' ignore it. ' is kept to retain negation
            if re.match('[A-Za-z\']+$', token.text) is None:
                continue
            pos_tags.append(self.pos2index(token.tag_))
            ner_tags.append(self.ner2index((token.ent_iob_ + '_' + token.ent_type_) if len(token.ent_type_) > 0 else token.ent_iob_))
            if token.like_num:
                text_tokens.append(self.word2index(u"NUM"))
            else:
                text_tokens.append(self.word2index(token.text))

        return text_tokens, pos_tags, ner_tags

    def ids2text(self, text_ids, decoder):
        text = []
        for text_id in text_ids:
            text.append(decoder[text_id])
        text = u' '.join(text)
        return text

    def write_to_file(self, texts, poss, ners):
        if self.to_file:
            with open(self.to_file, 'a', encoding = 'utf8') as fhw:
                fhw.write('\n#########################\n')
                for text_tokens, pos_tags, ner_tags in zip(texts, poss, ners):
                    fhw.write(self.ids2text(text_tokens, self.decoder))
                    fhw.write(u'\n-----------------\n')
                    fhw.write(self.ids2text(pos_tags, self.decoder_pos))
                    fhw.write(u'\n-----------------\n')
                    fhw.write(self.ids2text(ner_tags, self.decoder_ner))
                    fhw.write(u'\n-----------------\n')

    def encode_args(self, texts, max_arg_len = 100):
        texts_tokens = []
        poss_tags = []
        ners_tags = []
        for text in texts:
            # For ftfy.fix_text see https://ftfy.readthedocs.io/en/latest/
            text = self.nlp(ftfy.fix_text(clean(text)))
            # At this point text is of type spacy.tokens.doc.Doc
            text_tokens, pos_tags, ner_tags = self._encode_args(text, max_arg_len = max_arg_len)
            texts_tokens.append(text_tokens)
            poss_tags.append(pos_tags)
            ners_tags.append(ner_tags)

        self.write_to_file(texts_tokens, poss_tags, ners_tags)

        return (texts_tokens, poss_tags, ners_tags)

    def encode_wp(self, texts, max_wp_len = 500):

        texts_tokens = []
        poss_tags = []
        ners_tags = []
        for text in texts:
            # For ftfy.fix_text see https://ftfy.readthedocs.io/en/latest/
            text = self.nlp(ftfy.fix_text(clean(text)))
            text_tokens, pos_tags, ner_tags = self._encode_wp(text)
            texts_tokens.append(text_tokens)
            poss_tags.append(pos_tags)
            ners_tags.append(ner_tags)

        left_arg_words = texts_tokens[0]
        right_arg_words = texts_tokens[1]

        left_arg_pos = poss_tags[0]
        right_arg_pos = poss_tags[1]

        left_arg_ner = ners_tags[0]
        right_arg_ner = ners_tags[1]

        wp_len = len(left_arg_words) * len(right_arg_words)

        if wp_len > max_wp_len:
            arg_maxlen = int(math.floor(math.sqrt(max_wp_len)))
            if len(left_arg_words) <= len(right_arg_words):
                left_arg_words = left_arg_words[:arg_maxlen]
                left_arg_pos = left_arg_pos[:arg_maxlen]
                left_arg_ner = left_arg_ner[:arg_maxlen]
                right_arg_maxlen = max_wp_len // len(left_arg_words)
                right_arg_words = right_arg_words[:right_arg_maxlen]
                right_arg_pos = right_arg_pos[:right_arg_maxlen]
                right_arg_ner = right_arg_ner[:right_arg_maxlen]
            else:
                right_arg_words = right_arg_words[:arg_maxlen]
                right_arg_pos = right_arg_pos[:arg_maxlen]
                right_arg_ner = right_arg_ner[:arg_maxlen]
                left_arg_maxlen = max_wp_len // len(right_arg_words)
                left_arg_words = left_arg_words[:left_arg_maxlen]
                left_arg_pos = left_arg_pos[:left_arg_maxlen]
                left_arg_ner = left_arg_ner[:left_arg_maxlen]

        wp_len = len(left_arg_words) * len(right_arg_words)

        self.wplen_dist[wp_len] += 1

        self.write_to_file([left_arg_words, right_arg_words], [left_arg_pos, right_arg_pos], [left_arg_ner, right_arg_ner])

        return ([left_arg_words, right_arg_words], [left_arg_pos, right_arg_pos], [left_arg_ner, right_arg_ner])


def add_connective(arg2_text, connective):

    if len(connective) == 0:
        return arg2_text

    if arg2_text[0].islower():
        # if lower, add connective to start of the argument.
        arg2_text = u' '.join([connective, arg2_text])
    elif arg2_text[0].isupper():
        # if upper, then title the connective and then join them.
        arg2_words = arg2_text.split()
        # a lot of arguments start with Mr. or Ms. or some named entity like U.S etc, we don't want to lower case them
        if re.match('[A-Za-z\']+$', arg2_words[0]) is not None:
            arg2_text = u''.join([arg2_text[0].lower(), arg2_text[1:]])
        arg2_text = u' '.join([connective.title(), arg2_text])

    return arg2_text

