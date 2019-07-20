import cPickle
import itertools
import re

import lasagne
import numpy as np
import theano
import theano.tensor as T
from gensim.models import KeyedVectors
import pickle
from sklearn.metrics import f1_score, accuracy_score


def get_loss(loss_type, prediction, y):

    loss = None

    if loss_type == 'categorical_crossentropy':
        loss = lasagne.objectives.categorical_crossentropy(prediction, y)
    elif loss_type == 'multiclass_hinge_loss':
        loss = lasagne.objectives.multiclass_hinge_loss(prediction, y)

    return loss


def pad(X, indices):
    if X is None:
        return X
    X_xtra = X[indices]
    return np.append(X, X_xtra, axis = 0)


def pad_dataset(ds, batch_size):

    sz = len(ds[0])
    ds_new = []
    if sz % batch_size > 0:
        pad_sz = batch_size - (sz % batch_size)
        rnd_perm = np.random.permutation(range(sz))
        rnd_perm = rnd_perm[:pad_sz]
        for X in ds:
            ds_new.append(pad(X, rnd_perm))

    return ds_new, sz


def make_shared(X, cast_type, borrow = True):
    if X is None:
        return X
    X_shared = theano.shared(np.asarray(X, dtype = theano.config.floatX), borrow = borrow)
    X_shared = T.cast(X_shared, cast_type)
    return X_shared


def shared_dataset(data, borrow = True):

    data_shared = []
    for X in data[:-1]:
        data_shared.append(make_shared(X, 'int64', borrow = borrow))
    data_shared.append(make_shared(data[-1], 'int32', borrow = borrow))
    return data_shared


def save_model(filename, param_values):
    f = file(filename, 'wb')
    cPickle.dump(param_values, f, protocol = cPickle.HIGHEST_PROTOCOL)
    f.close()

# def save_model(filename, param_values):
#     np.savez(filename, *param_values)


def load_model(filename):
    f = file(filename, 'rb')
    param_values = cPickle.load(f)
    f.close()
    return param_values

# def load_model(filename):
#     with np.load(filename) as f:
#         param_values = [f['arr_%d' % i] for i in range(len(f.files))]
#     return param_values


def get_time_stamp(file_name):

    ts_regex = re.compile('\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}')

    timestamps = ts_regex.findall(file_name)

    timestamp = ''

    if len(timestamps) > 0:
        timestamp = timestamps[0]

    return timestamp


def get_class_weights(class_dict, examples_per_class):

    total = 0.0
    for _class in examples_per_class:
        total += (float(1) / examples_per_class[_class])

    K = float(1) / total

    inv_class_dict = {v: k for k, v in class_dict.iteritems()}

    n_classes = len(inv_class_dict)
    class_weights = []
    for i in xrange(n_classes):
        class_weights.append(K / examples_per_class[inv_class_dict[i]])

    return class_weights


def get_class_names(class_dict):

    class_names = []
    inv_class_dict = {v: k for k, v in class_dict.iteritems()}
    for i in xrange(len(inv_class_dict)):
        class_names.append(inv_class_dict[i])
    return class_names


def compute_metrices(y_true, y_pred, binary_class = 1, binarize = False):

    if binarize:
        test_acc = accuracy_score(y_true, y_pred)
        test_f1_macro = f1_score(y_true, y_pred, average = 'binary', pos_label = binary_class)
        test_f1_micro = test_f1_macro
    else:
        test_acc = accuracy_score(y_true, y_pred)
        test_f1_macro = f1_score(y_true, y_pred, average = 'macro')
        test_f1_micro = f1_score(y_true, y_pred, average = 'micro')

    return test_acc, test_f1_macro, test_f1_micro


def post_process_predictions(y, y_pred, multilabel, binary_class, binarize):

    _y = []
    _y_pred = []
    for ii in xrange(len(y)):
        y_c = y[ii]
        y_pred_c = y_pred[ii]
        if multilabel and (y_pred_c in y_c):
            _y.append(y_pred_c)
            _y_pred.append(y_pred_c)
        else:
            _y.append(y_c[0])
            _y_pred.append(y_pred_c)

    return _y, _y_pred


def print_params(params):

    if isinstance(params, dict):
        params_dict = params
    else:
        params_dict = vars(params)

    for k, v in params_dict.iteritems():
        print k, v


def get_idx_from_word_pairs(left_words, right_words, max_l, max_filter_len, stride = 2):
    """
    Transforms sentence into a list of indices.
    At this point there is a need to pad zeros to make all sentences of same length
    We have max_l word pairs so 2 * max_l words and then we pad atleast
    (max_filter_len - stride) before and after the word pairs
    """

    wp = []
    for pair in itertools.product(left_words, right_words):
        wp.append(pair)

    x = []
    pad = max_filter_len - stride

    for _ in range(pad):
        x.append(0)

    for word_pair in wp:
        w1, w2 = word_pair
        x.append(w1)
        x.append(w2)

    while len(x) < (2 * max_l + 2 * pad):
        x.append(0)

    return np.asarray(x)


def get_idx_from_argument(argument, max_l, max_filter_len, stride = 1):
    """
    Transforms sentence into a list of indices.
    At this point there is a need to pad zeros to make all sentences of same length
    We have max_l words and then we pad atleast
    (max_filter_len - stride) before and after the words
    """

    x = []
    pad = max_filter_len - stride

    for _ in range(pad):
        x.append(0)

    for w in argument:
        x.append(w)

    while len(x) < (max_l + 2 * pad):
        x.append(0)

    return np.asarray(x)


def make_idx_data(arguments, max_arg_len, max_wp_len, max_arg_fsz, max_wp_fsz):
    """
    Transforms sentences into a 2-d matrix.
    """
    X_larg = []
    X_rarg = []
    X_lpos = []
    X_rpos = []
    X_lner = []
    X_rner = []
    X_wp = []
    X_wp_rev = []
    X_wp_pos = []
    X_wp_rev_pos = []
    X_wp_ner = []
    X_wp_rev_ner = []
    y = []

    for argument in arguments:
        X_larg.append(get_idx_from_argument(argument["left_arg_words"], max_arg_len, max_arg_fsz))
        X_rarg.append(get_idx_from_argument(argument["right_arg_words"], max_arg_len, max_arg_fsz))
        X_lpos.append(get_idx_from_argument(argument["left_arg_pos"], max_arg_len, max_arg_fsz))
        X_rpos.append(get_idx_from_argument(argument["right_arg_pos"], max_arg_len, max_arg_fsz))
        X_lner.append(get_idx_from_argument(argument["left_arg_ner"], max_arg_len, max_arg_fsz))
        X_rner.append(get_idx_from_argument(argument["right_arg_ner"], max_arg_len, max_arg_fsz))

        X_wp.append(get_idx_from_word_pairs(argument["left_wp_words"], argument["right_wp_words"], max_wp_len, max_wp_fsz))
        X_wp_rev.append(get_idx_from_word_pairs(argument["right_wp_words"], argument["left_wp_words"], max_wp_len, max_wp_fsz))
        X_wp_pos.append(get_idx_from_word_pairs(argument["left_wp_pos"], argument["right_wp_pos"], max_wp_len, max_wp_fsz))
        X_wp_rev_pos.append(get_idx_from_word_pairs(argument["right_wp_pos"], argument["left_wp_pos"], max_wp_len, max_wp_fsz))
        X_wp_ner.append(get_idx_from_word_pairs(argument["left_wp_ner"], argument["right_wp_ner"], max_wp_len, max_wp_fsz))
        X_wp_rev_ner.append(get_idx_from_word_pairs(argument["right_wp_ner"], argument["left_wp_ner"], max_wp_len, max_wp_fsz))
        y.append(argument["y"])  # At this point, argument["y"] is a list

    X_larg = np.asarray(X_larg)
    X_rarg = np.asarray(X_rarg)
    X_lpos = np.asarray(X_lpos)
    X_rpos = np.asarray(X_rpos)
    X_lner = np.asarray(X_lner)
    X_rner = np.asarray(X_rner)

    X_wp = np.asarray(X_wp)
    X_wp_rev = np.asarray(X_wp_rev)
    X_wp_pos = np.asarray(X_wp_pos)
    X_wp_rev_pos = np.asarray(X_wp_rev_pos)
    X_wp_ner = np.asarray(X_wp_ner)
    X_wp_rev_ner = np.asarray(X_wp_rev_ner)
    y = np.asarray(y)

    dataset = [X_larg, X_rarg, X_lpos, X_rpos, X_lner, X_rner, X_wp, X_wp_rev, X_wp_pos, X_wp_rev_pos, X_wp_ner, X_wp_rev_ner, y]
    return dataset


def load_data(input_file, max_arg_fsz, max_wp_fsz):
    '''
    max_arg_fsz - max filter size for arguments
    max_wp_fsz - max filter size for word pairs
    '''

    print("loading data...")
    x = pickle.load(open(input_file, "rb"))

    _train_data, _val_data, _test_data, class_dict, examples_per_class, max_arg_len, max_wp_len = x[0], x[1], x[2], x[3], x[4], x[5], x[6]

    train_data = make_idx_data(_train_data, max_arg_len, max_wp_len, max_arg_fsz, max_wp_fsz)
    val_data = make_idx_data(_val_data, max_arg_len, max_wp_len, max_arg_fsz, max_wp_fsz)
    test_data = make_idx_data(_test_data, max_arg_len, max_wp_len, max_arg_fsz, max_wp_fsz)

    return train_data, val_data, test_data, class_dict, examples_per_class


def make_idx_data_for_mtl(arguments, max_arg_len, max_wp_len, max_arg_fsz, max_wp_fsz):
    """
    Transforms sentences into a 2-d matrix.
    """
    X_larg = []
    X_rarg = []
    X_lpos = []
    X_rpos = []
    X_lner = []
    X_rner = []
    X_wp = []
    X_wp_rev = []
    X_wp_pos = []
    X_wp_rev_pos = []
    X_wp_ner = []
    X_wp_rev_ner = []
    y = []
    is_imp = []

    for argument in arguments:
        X_larg.append(get_idx_from_argument(argument["left_arg_words"], max_arg_len, max_arg_fsz))
        X_rarg.append(get_idx_from_argument(argument["right_arg_words"], max_arg_len, max_arg_fsz))
        X_lpos.append(get_idx_from_argument(argument["left_arg_pos"], max_arg_len, max_arg_fsz))
        X_rpos.append(get_idx_from_argument(argument["right_arg_pos"], max_arg_len, max_arg_fsz))
        X_lner.append(get_idx_from_argument(argument["left_arg_ner"], max_arg_len, max_arg_fsz))
        X_rner.append(get_idx_from_argument(argument["right_arg_ner"], max_arg_len, max_arg_fsz))

        X_wp.append(get_idx_from_word_pairs(argument["left_wp_words"], argument["right_wp_words"], max_wp_len, max_wp_fsz))
        X_wp_rev.append(get_idx_from_word_pairs(argument["right_wp_words"], argument["left_wp_words"], max_wp_len, max_wp_fsz))
        X_wp_pos.append(get_idx_from_word_pairs(argument["left_wp_pos"], argument["right_wp_pos"], max_wp_len, max_wp_fsz))
        X_wp_rev_pos.append(get_idx_from_word_pairs(argument["right_wp_pos"], argument["left_wp_pos"], max_wp_len, max_wp_fsz))
        X_wp_ner.append(get_idx_from_word_pairs(argument["left_wp_ner"], argument["right_wp_ner"], max_wp_len, max_wp_fsz))
        X_wp_rev_ner.append(get_idx_from_word_pairs(argument["right_wp_ner"], argument["left_wp_ner"], max_wp_len, max_wp_fsz))
        is_imp.append(argument["is_imp"])
        y.append(argument["y"])  # At this point, argument["y"] is a list

    X_larg = np.asarray(X_larg)
    X_rarg = np.asarray(X_rarg)
    X_lpos = np.asarray(X_lpos)
    X_rpos = np.asarray(X_rpos)
    X_lner = np.asarray(X_lner)
    X_rner = np.asarray(X_rner)

    X_wp = np.asarray(X_wp)
    X_wp_rev = np.asarray(X_wp_rev)
    X_wp_pos = np.asarray(X_wp_pos)
    X_wp_rev_pos = np.asarray(X_wp_rev_pos)
    X_wp_ner = np.asarray(X_wp_ner)
    X_wp_rev_ner = np.asarray(X_wp_rev_ner)
    is_imp = np.asarray(is_imp)
    y = np.asarray(y)
    dataset = [X_larg, X_rarg, X_lpos, X_rpos, X_lner, X_rner, X_wp, X_wp_rev, X_wp_pos, X_wp_rev_pos, X_wp_ner, X_wp_rev_ner, is_imp, y]
    return dataset


def load_data_for_mtl(input_file, max_arg_fsz, max_wp_fsz):
    '''
    max_arg_fsz - max filter size for arguments
    max_wp_fsz - max filter size for word pairs
    '''

    print("loading data...")
    x = pickle.load(open(input_file, "rb"))

    _train_data, _val_data, _test_data, class_dict, examples_per_class, max_arg_len, max_wp_len = x[0], x[1], x[2], x[3], x[4], x[5], x[6]

    train_data = make_idx_data_for_mtl(_train_data, max_arg_len, max_wp_len, max_arg_fsz, max_wp_fsz)
    val_data = make_idx_data_for_mtl(_val_data, max_arg_len, max_wp_len, max_arg_fsz, max_wp_fsz)
    test_data = make_idx_data_for_mtl(_test_data, max_arg_len, max_wp_len, max_arg_fsz, max_wp_fsz)

    return train_data, val_data, test_data, class_dict, examples_per_class


def load_w2v(embedding_file):
    if not embedding_file:
        return None
    word2vec = KeyedVectors.load_word2vec_format(embedding_file, binary = True)
    print('Found %s word vectors of word2vec' % len(word2vec.vocab))
    return word2vec


def get_W(w2v_file, word2idx, init_embs_file = None, mod_embs_file = None, w2i_index = None, use_modified_embs = False, k = 300):

    total_misses_mod_embs = 0
    total_misses_w2v = 0

    word2vec = load_w2v(w2v_file)

    mod_embs = None
    mod_embs_word2idx = None

#     if use_modified_embs:
#         mod_embs, mod_embs_word2idx = load_modified_embs(init_embs_file, mod_embs_file, w2i_index)

    W = np.zeros(shape = (len(word2idx), k), dtype = 'float32')

    for word in word2idx:
        if mod_embs_word2idx and word in mod_embs_word2idx:
            W[word2idx[word]] = mod_embs[mod_embs_word2idx[word]]
        elif word2vec and (word in word2vec.vocab):
            W[word2idx[word]] = word2vec.word_vec(word)
            total_misses_mod_embs += 1
        else:
            W[word2idx[word]] = np.random.uniform(-0.25, 0.25, k)
            total_misses_w2v += 1
            total_misses_mod_embs += 1

    W[0] = np.zeros(k, dtype = 'float32')

    print ('Total number of misses in modified embeddings: ', total_misses_mod_embs)
    print ('Total number of misses in w2v: ', total_misses_w2v)

    return W


def get_binary_class_index(class_dict):
    for cname in class_dict.keys():
        if cname != 'Other':
            return class_dict[cname]
    return None


def tune(y_proba, y_true, pos_label):
    thresholds = np.arange(0.2, 0.8, 0.05)
    # filter out examples which have multiple labels. We don't want to include them for threshold tunning
    tmp = [(y_proba_c, y_true_c) for y_proba_c, y_true_c in zip(y_proba, y_true) if len(y_true_c) == 1]
    y_proba_new, y_true_new = zip(*tmp)
    y_proba_new = np.asarray(list(y_proba_new))
    y_proba_new = y_proba_new[:, pos_label]
    y_true_new = np.squeeze(np.asarray(list(y_true_new)))

    fscores = []
    for threshold in thresholds:
        y_pred = np.where(y_proba_new > threshold, pos_label, 1 - pos_label)
        fscore = f1_score(y_true_new, y_pred, average = 'binary', pos_label = pos_label)
        fscores.append(fscore)
    best_threshold = thresholds[np.argmax(fscores)]
    return best_threshold


def post_process(bt, proba, y_true, multilabel, binarize, pos_label):

        if binarize:
            if bt == -1:
                # threshold tunning is not helping so no need to tune the threshold
                # bt = tune(proba, y_true, pos_label)
                bt = 0.5
            proba = proba[:, pos_label]
            y_pred = np.where(proba > bt, pos_label, 1 - pos_label)
        else:
            y_pred = np.argmax(proba, axis = 1)

        y_true, y_pred = post_process_predictions(y_true, y_pred, multilabel, binary_class = pos_label, binarize = binarize)

        return bt, y_true, y_pred
