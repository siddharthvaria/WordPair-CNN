import argparse
from collections import OrderedDict
from collections import defaultdict
import datetime
from common.text_utils import SubwordEncoder, WordEncoder, add_connective
import os
import pickle
import numpy as np
import json


class Rel_Preprocessor(object):

    def __init__(self, args):

        self.rel_type = args['rel_type']
        if self.rel_type == 'imp':
            self.rel_type = 'Implicit'
        elif self.rel_type == 'exp':
            self.rel_type = 'Explicit'
        else:
            raise (ValueError, "I can't cope with this value!")

        self.task_type = args['task_type']

        self.replicate = args['replicate']
        self.binarize = args['binarize']
        self.binary_class = self.prefix_to_class(args['binary_class'])
        self.augment = args['augment']
        if args['augment']:
            self.omissible_markers = self.read_file(args['omissible_markers_file'])

        self.train_dataset = []
        self.dev_dataset = []
        self.test_dataset = []
        self.class2idx = {}

        self.examples_per_class = defaultdict(int)
        self.imp_examples_per_class = defaultdict(int)
        self.exp_examples_per_class = defaultdict(int)
        self.tr_examples_per_class = defaultdict(int)
        self.imp_tr_examples_per_class = defaultdict(int)
        self.exp_tr_examples_per_class = defaultdict(int)
        self.te_examples_per_class = defaultdict(int)
        self.imp_te_examples_per_class = defaultdict(int)
        self.exp_te_examples_per_class = defaultdict(int)
        self.val_examples_per_class = defaultdict(int)
        self.imp_val_examples_per_class = defaultdict(int)
        self.exp_val_examples_per_class = defaultdict(int)
        self.total_examples = 0

        self.token_type = args['token_type']
        # self.stop_words = self.read_stopwords(args['stopwords_file'])
        if args['token_type'] == 'word':
            self.text_encoder = WordEncoder(to_file = args['to_file'])
        else:
            self.text_encoder = SubwordEncoder(args['encoder_path'], args['bpe_path'], to_file = args['to_file'])

        # self.left_arg_lens = defaultdict(int)
        # self.right_arg_lens = defaultdict(int)

        self.class2connective = defaultdict(set)

        self.include_connective = args['include_connective']

        self.max_arg_len = args['max_arg_len']
        self.max_wp_len = args['max_wp_len']

        self.blind_test_set_file = args['blind_test_set_file']

    def prefix_to_class(self, prefix):

        if prefix == 'com':
            return 'Comparison'
        elif prefix == 'con':
            return 'Contingency'
        elif prefix == 'exp':
            return 'Expansion'
        else:
            return 'Temporal'

    def read_file(self, input_file):
        if not input_file:
            return None
        words = []
        with open(input_file, 'r', encoding = 'utf8') as fhr:
            for line in fhr:
                line = line.strip()
                words.append(line)
        return words

    def initialize_classes(self):
        if self.binarize:
            # binary classification
            class_names = [self.binary_class, 'Other']
        else:
            # multi class classification
            if self.task_type == '4_way':
                class_names = ['Contingency', 'Comparison', 'Expansion', 'Temporal']
            elif self.task_type == '15_way':
                class_names = ['Temporal.Synchrony', 'Temporal.Asynchronous.Precedence',
                                'Temporal.Asynchronous.Succession', 'Contingency.Cause.Reason',
                                'Contingency.Cause.Result', 'Contingency.Condition', 'Comparison.Contrast',
                                 'Comparison.Concession', 'Expansion.Conjunction', 'Expansion.Instantiation',
                                 'Expansion.Restatement', 'Expansion.Alternative', 'Expansion.Alternative.Chosen_alternative',
                                 'Expansion.Exception', 'EntRel']
            elif self.task_type == '11_way':
                class_names = ['Temporal.Synchrony', 'Temporal.Asynchronous', 'Contingency.Cause',
                               'Contingency.Pragmatic_cause', 'Comparison.Contrast', 'Comparison.Concession',
                               'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.Restatement',
                               'Expansion.Alternative', 'Expansion.List']

        class_names.sort()

        cid = 0
        for class_name in class_names:
            self.class2idx[class_name] = cid
            cid += 1

        self.idx2class = {v:k for k, v in self.class2idx.items()}

    def parse_class_names(self, class_names, marker):

        rel_classes = []
        for rel_class in class_names:
            self.class2connective[rel_class].add(marker)
            current_class = self.parse_class_name(rel_class)
            if not current_class in self.class2idx:
                # raise ValueError('Bad class name!')
                continue
            rel_classes.append(self.class2idx[current_class])
        return rel_classes

    def parse_class_name(self, rel_class):
        if self.task_type == '15_way':
            cnames = ['Contingency.Cause.Reason', 'Contingency.Pragmatic_cause']
            for cname in cnames:
                if cname in rel_class:
                    return 'Contingency.Cause.Reason'
            cnames = ['Contingency.Condition', 'Contingency.Pragmatic_condition']
            for cname in cnames:
                if cname in rel_class:
                    return 'Contingency.Condition'
            cnames = ['Comparison.Contrast', 'Comparison.Pragmatic_contrast']
            for cname in cnames:
                if cname in rel_class:
                    return 'Comparison.Contrast'
            cnames = ['Comparison.Concession', 'Comparison.Pragmatic_concession']
            for cname in cnames:
                if cname in rel_class:
                    return 'Comparison.Concession'
            cnames = ['Expansion.Conjunction', 'Expansion.List']
            for cname in cnames:
                if cname in rel_class:
                    return 'Expansion.Conjunction'
            cnames = ['Expansion.Restatement']
            for cname in cnames:
                if cname in rel_class:
                    return 'Expansion.Restatement'
            cnames = ['Expansion.Alternative.Conjunctive', 'Expansion.Alternative.Disjunctive']
            for cname in cnames:
                if cname in rel_class:
                    return 'Expansion.Alternative'
            # all others
            cnames = ['Temporal.Synchrony', 'Temporal.Asynchronous', 'Temporal.Asynchronous.Precedence',
                      'Temporal.Asynchronous.Succession', 'Contingency.Cause', 'Contingency.Cause.Result', 'Expansion.Instantiation',
                      'Expansion.Alternative', 'Expansion.Alternative.Chosen_alternative',
                      'Expansion.Exception', 'EntRel', 'Expansion', 'Temporal', 'Comparison', 'Contingency']
            for cname in cnames:
                if cname == rel_class:
                    return rel_class
            return None
        elif self.task_type == '11_way':
            rel_class_splits = rel_class.split('.')
            if len(rel_class_splits) >= 2:
                return '.'.join(rel_class_splits[:2])
            return rel_class
        else:
            try:
                dot_index = rel_class.index('.')
            except ValueError:
                dot_index = -1
            if dot_index >= 0:
                current_class = rel_class[:dot_index]
            else:
                current_class = rel_class

            if self.binarize:
                if current_class != self.binary_class:
                    current_class = 'Other'

            return current_class

    def get_y(self, y_c, y_all):
        y = []
        y.append(y_c)
        for y_i in y_all:
            if y_i == y_c:
                continue
            y.append(y_i)
        assert len(y) == len(y_all), 'In method get_y, len(y) != len(y_all)'
        return y

    def remove_duplicates(self, arr):
        od = OrderedDict()
        for v in arr:
            od[v] = ''

        return od.keys()

    def process_dataset(self, dataset, dataset_type):

        for document in dataset:

            for relation in document['relations']:

                # If task is 4_way then we want either explicit or implicit
                # If task is 15_way then we want everything except NoRel

                if self.task_type == '4_way' and relation['type'] not in ['Explicit', 'Implicit']:
                    continue

                if self.augment:
                    if self.task_type == '4_way' and relation['type'] == 'Explicit':
                        if relation['marker'] not in self.omissible_markers:
                            continue

                if self.task_type == '11_way' and relation['type'] not in ['Explicit', 'Implicit']:
                    continue

                if self.task_type == '15_way' and relation['type'] == 'NoRel':
                    continue

                # drop all cases where the arguments have more than one relation between them.
                # if len(relation['class']) != 1:
                #    continue

                # raw_arg1 = relation['arg1']['text_span']['text']
                # raw_arg2 = relation['arg2']['text_span']['text']

                if self.include_connective:
                    raw_arg1 = relation['arg1']['text_span']['text']
                    if 'text_wc' in relation['arg2']['text_span']:
                        raw_arg2 = relation['arg2']['text_span']['text_wc']
                    else:
                        raw_arg2 = relation['arg2']['text_span']['text']
                else:
                    raw_arg1 = relation['arg1']['text_span']['text']
                    raw_arg2 = relation['arg2']['text_span']['text']

                # connective and label need to be added to datum
                datum = self.get_datum(raw_arg1, raw_arg2, max_arg_len = self.max_arg_len, max_wp_len = self.max_wp_len)

                if datum["left_arg_words"] == None or datum["right_arg_words"] == None:
                    continue

                rel_classes = self.parse_class_names(relation['class'], relation['marker'])

                # there might be duplicates in binary case. remove them not disturbing the order otherwise it will mess up with class counts
                rel_classes = self.remove_duplicates(rel_classes)

                for rel_class in rel_classes:
                    curr_datum = {"y": self.get_y(rel_class, rel_classes),
                                  "connective": relation['marker'],
                                  "is_imp": 0 if relation['type'] == 'Explicit' else 1}
                    is_imp = curr_datum["is_imp"]
                    curr_datum.update(datum)
                    if dataset_type == 'train':
                        self.tr_examples_per_class[self.idx2class[rel_class]] += 1
                        if is_imp == 1:
                            self.imp_tr_examples_per_class[self.idx2class[rel_class]] += 1
                        else:
                            self.exp_tr_examples_per_class[self.idx2class[rel_class]] += 1
                        self.train_dataset.append(curr_datum)
                    elif dataset_type == 'test':
                        self.te_examples_per_class[self.idx2class[rel_class]] += 1
                        if is_imp == 1:
                            self.imp_te_examples_per_class[self.idx2class[rel_class]] += 1
                        else:
                            self.exp_te_examples_per_class[self.idx2class[rel_class]] += 1
                        self.test_dataset.append(curr_datum)
                    elif dataset_type == 'dev':
                        self.val_examples_per_class[self.idx2class[rel_class]] += 1
                        if is_imp == 1:
                            self.imp_val_examples_per_class[self.idx2class[rel_class]] += 1
                        else:
                            self.exp_val_examples_per_class[self.idx2class[rel_class]] += 1
                        self.dev_dataset.append(curr_datum)

                    self.examples_per_class[self.idx2class[rel_class]] += 1
                    if is_imp == 1:
                        self.imp_examples_per_class[self.idx2class[rel_class]] += 1
                    else:
                        self.exp_examples_per_class[self.idx2class[rel_class]] += 1

                    self.total_examples += 1

                    if (dataset_type == 'test') or (dataset_type == 'dev') or not self.replicate:
                        break

    def resample_data(self):

        self.train_dataset = np.asarray(self.train_dataset)
        self.dev_dataset = np.asarray(self.dev_dataset)
        self.test_dataset = np.asarray(self.test_dataset)

        if not self.binarize:
            return

        _binary_ds = []
        _other_ds = []
        for ii in range(len(self.train_dataset)):
            dp = self.train_dataset[ii]
            if dp['y'][0] == self.class2idx[self.binary_class]:
                _binary_ds.append(dp)
            else:
                _other_ds.append(dp)

        _other_ds = np.asarray(_other_ds)
        _binary_ds = np.asarray(_binary_ds)

        rnd_perm = np.random.permutation(range(len(_other_ds)))
        rnd_perm = rnd_perm[:min(len(_other_ds), len(_binary_ds))]
        _other_ds = _other_ds[rnd_perm]

        self.tr_examples_per_class[self.binary_class] = len(_binary_ds)
        self.tr_examples_per_class['Other'] = len(_other_ds)

        for k in [self.binary_class, 'Other']:
            self.examples_per_class[k] = self.tr_examples_per_class[k]
            self.examples_per_class[k] += self.val_examples_per_class[k]
            self.examples_per_class[k] += self.te_examples_per_class[k]

        _binary_ds = np.concatenate((_binary_ds, _other_ds), axis = 0)
        rnd_perm = np.random.permutation(range(len(_binary_ds)))
        _binary_ds = _binary_ds[rnd_perm]
        self.train_dataset = _binary_ds

    def build_data(self, input_file):

        train_set, dev_set, test_set = pickle.load(open(input_file, "rb"))

        # initialize classes for 4_way or 15_way classification tasks
        self.initialize_classes()

        print ('Processing train set . . .')
        self.process_dataset(train_set, 'train')
        print ('Processing dev set . . .')
        self.process_dataset(dev_set, 'dev')
        print ('Processing test set . . .')
        if self.blind_test_set_file:
            blind_test_set = get_blind_test_set(self.blind_test_set_file)
            self.process_dataset(blind_test_set, 'test')
        else:
            self.process_dataset(test_set, 'test')

        print ('Total number of examples: ', self.total_examples)
        print ('Total number of train examples: ', len(self.train_dataset))
        print ('Total number of dev examples: ', len(self.dev_dataset))
        print ('Total number of test examples: ', len(self.test_dataset))

        print ('Total examples per class:')
        print ('-------------------------')
        for k, v in self.examples_per_class.items():
            print (k, v)
        print ('-------------------------')

        print ('Implicit total examples per class:')
        print ('-------------------------')
        for k, v in self.imp_examples_per_class.items():
            print (k, v)
        print ('-------------------------')

        print ('Explicit total examples per class:')
        print ('-------------------------')
        for k, v in self.exp_examples_per_class.items():
            print (k, v)
        print ('-------------------------')

        print ('Training examples per class:')
        print ('-------------------------')
        for k, v in self.tr_examples_per_class.items():
            print (k, v)
        print ('-------------------------')

        print ('Implicit training examples per class:')
        print ('-------------------------')
        for k, v in self.imp_tr_examples_per_class.items():
            print (k, v)
        print ('-------------------------')

        print ('Explicit training examples per class:')
        print ('-------------------------')
        for k, v in self.exp_tr_examples_per_class.items():
            print (k, v)
        print ('-------------------------')

        print ('Validation examples per class:')
        print ('-------------------------')
        for k, v in self.val_examples_per_class.items():
            print (k, v)
        print ('-------------------------')

        print ('Implicit validation examples per class:')
        print ('-------------------------')
        for k, v in self.imp_val_examples_per_class.items():
            print (k, v)
        print ('-------------------------')

        print ('Explicit validation examples per class:')
        print ('-------------------------')
        for k, v in self.exp_val_examples_per_class.items():
            print (k, v)
        print ('-------------------------')

        print ('Test examples per class:')
        print ('-------------------------')
        for k, v in self.te_examples_per_class.items():
            print (k, v)
        print ('-------------------------')

        print ('Implicit test examples per class:')
        print ('-------------------------')
        for k, v in self.imp_te_examples_per_class.items():
            print (k, v)
        print ('-------------------------')

        print ('Explicit test examples per class:')
        print ('-------------------------')
        for k, v in self.exp_te_examples_per_class.items():
            print (k, v)
        print ('-------------------------')

#         self.left_arg_lens = OrderedDict(sorted(self.left_arg_lens.items(), key = itemgetter(0)))
#         print('Left argument length distribution:')
#         print ('-------------------------')
#         for k, v in self.left_arg_lens.items():
#             print (k, v)
#         print ('-------------------------')
#
#         self.right_arg_lens = OrderedDict(sorted(self.right_arg_lens.items(), key = itemgetter(0)))
#         print('Right argument length distribution:')
#         print ('-------------------------')
#         for k, v in self.right_arg_lens.items():
#             print (k, v)
#         print ('-------------------------')

        print('class2connective:')
        print ('-------------------------')
        for k, v in self.class2connective.items():
            print('class: ', k)
            for _v in v:
                print (_v)
        print ('-------------------------')

        print('Vocabulary size: ', len(self.text_encoder.encoder))

    def get_datum(self, l_arg, r_arg, max_arg_len = 100, max_wp_len = 500):
        datum = {}
        datum["left_original"] = l_arg
        datum["right_original"] = r_arg
        if self.token_type == 'word':
            texts_tokens, poss_tags, ners_tags = self.text_encoder.encode_args([l_arg, r_arg], max_arg_len = max_arg_len)
            datum["left_arg_words"] = texts_tokens[0]
            datum["right_arg_words"] = texts_tokens[1]
            datum["left_arg_pos"] = poss_tags[0]
            datum["right_arg_pos"] = poss_tags[1]
            datum["left_arg_ner"] = ners_tags[0]
            datum["right_arg_ner"] = ners_tags[1]
            texts_tokens, poss_tags, ners_tags = self.text_encoder.encode_wp([l_arg, r_arg], max_wp_len = max_wp_len)
            datum["left_wp_words"] = texts_tokens[0]
            datum["right_wp_words"] = texts_tokens[1]
            datum["left_wp_pos"] = poss_tags[0]
            datum["right_wp_pos"] = poss_tags[1]
            datum["left_wp_ner"] = ners_tags[0]
            datum["right_wp_ner"] = ners_tags[1]
        else:
            texts_tokens = self.text_encoder.encode([[l_arg, r_arg]])
            datum["left_arg_words"] = texts_tokens[0]
            datum["right_arg_words"] = texts_tokens[1]
        return datum


def get_blind_test_set(blind_test_set_file):

#     print(u'arg1: ', relation[u'arg1'][u'text_span'][u'text'])
#     print(u'arg2: ', relation[u'arg2'][u'text_span'][u'text'])
#     relation['arg2']['text_span']['text_wc']
#     print(u'classes: ', relation[u'class'])
#     relation[u'marker']
#     relation[u'type']

    doc = {}
    doc[u'relations'] = []
    with open(blind_test_set_file, 'r') as fhr:
        for line in fhr:
            line = line.strip()
            datum = json.loads(line)
            # datum["Arg1"]["RawText"]
            # datum["Arg1"]["RawText"]
            # datum["Connective"]"RawText"
            # datum["Sense"]: ["Temporal.Asynchronous.Succession"],
            # datum["Type": "Explicit"]
            relation = {}
            relation['type'] = datum['Type']
            relation['marker'] = datum['Connective']['RawText']
            relation['class'] = datum['Sense']
            relation['arg1'] = {}
            relation['arg1']['text_span'] = {}
            relation['arg2'] = {}
            relation['arg2']['text_span'] = {}
            relation['arg1']['text_span']['text'] = datum["Arg1"]["RawText"]
            relation['arg2']['text_span']['text'] = datum["Arg2"]["RawText"]
            relation['arg2']['text_span']['text_wc'] = add_connective(datum["Arg2"]["RawText"], datum['Connective']['RawText'])
            doc[u'relations'].append(relation)

    return [doc]


def main(args):

    print('Command line arguments: ')
    for k, v in args.items():
        print(k, v)

    if args['binarize']:
        assert (args['binary_class'] in ['com', 'con', 'exp', 'tem']), '\'binary_class\' not in [com,con,exp,tem]'

    # pp = Rel_Preprocessor(args['rel_type'], args['task_type'], args['replicate'], args['binarize'], args['binary_class'])
    pp = Rel_Preprocessor(args)

    # pp.load_stop_words(args['stopwords_file'])

    pp.build_data(args['parsed_pdtb_file'])

    # To load the pre-trained embeddings from Task1, use w2i_index of 1
    # To load the pre-trained embeddings from Task2, use w2i_index of 4

    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    encoder_file_name = os.path.join(args['output_dir'], '{0}_{1}.p'.format(pp.text_encoder.__class__.__name__, time_stamp))
    pickle.dump([pp.text_encoder.encoder, pp.text_encoder.encoder_pos, pp.text_encoder.encoder_ner], open(encoder_file_name, "wb"), protocol = 2)

    pdtb_classifier_input_file_name = os.path.join(args['output_dir'], 'PDTB_Sentences_{0}_{1}.p'.format(pp.rel_type.lower(), time_stamp))

    pickle.dump([pp.train_dataset, pp.dev_dataset, pp.test_dataset, pp.class2idx, pp.tr_examples_per_class, pp.max_arg_len, pp.max_wp_len], open(pdtb_classifier_input_file_name, "wb"), protocol = 2)


def parse_cmd_args():

    parser = argparse.ArgumentParser(description = '')

    parser.add_argument('parsed_pdtb_file', help = 'output of parse_pdtb.py')  # output of parse_pdtb.py
    parser.add_argument('output_dir', help = 'output directory where the output files should be saved')
    parser.add_argument('-tat', '--task_type', default = '4_way', type = str, choices = ['4_way', '11_way', '15_way'], help = 'Task type is 4_way, 11_way or 15_way')
    parser.add_argument('-r', '--replicate', default = False, action = 'store_true', help = 'If True, training examples will be replicated once for each relation class')
    parser.add_argument('-b', '--binarize', default = False, action = 'store_true', help = 'If True, data set will be binarized based on \'binary_class\' argument')
    parser.add_argument('-bc', '--binary_class', default = 'com', type = str, choices = ['com', 'con', 'exp', 'tem'], help = 'If binarize is true, this argument specifies the positive class i.e one of \'com\',\'con\',\'exp\',\'tem\'')
    parser.add_argument('-rt', '--rel_type', default = 'imp', type = str, choices = ['imp', 'exp'], help = '\'imp\' or \'exp\' (Used only for 4_way task for now)')
    parser.add_argument('-tot', '--token_type', default = 'word', choices = ['word', 'subword'], help = 'Choose whether to preprocess text at word or subword level')
    parser.add_argument('-bp', '--bpe_path', default = None, type = str, help = 'bpe file path')
    parser.add_argument('-ep', '--encoder_path', default = None, type = str, help = 'vocabulary file path')
    parser.add_argument('-tf', '--to_file', default = None, type = str, help = 'To output preprocessed text to a file')
    parser.add_argument('-ma', '--max_arg_len', default = 100, type = int, help = 'maximum argument length in words')
    parser.add_argument('-mw', '--max_wp_len', default = 500, type = int, help = 'maximum word pairs length in words')
    parser.add_argument('-a', '--augment', default = False, action = 'store_true', help = 'augment implicit relations with explicit relations containing omissible markers')
    parser.add_argument('-omf', '--omissible_markers_file', default = None, type = str, help = 'path to freely omissible markers file')
    parser.add_argument('-ic', '--include-connective', default = False, action = 'store_true', help = 'Whether to prepend connective to explicit arg2 or not')
    # parser.add_argument('-sf', '--stopwords_file', default = None, help = 'path to stop words file')  # stopwords file
    parser.add_argument('-bts', '--blind_test_set_file', default = None, type = str, help = 'blind test set file path')

    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':

    main(parse_cmd_args())
