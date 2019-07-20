import argparse
import time
import math
import os
import pickle
import numpy as np
np.random.seed(57697)
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

from pytorch_impl.cnn_pdtb_classifier_utils_pytorch import get_W, post_process, print_params, get_time_stamp
from pytorch_impl.cnn_pdtb_classifier_utils_pytorch import get_class_weights, get_class_names, get_binary_class_index
from pytorch_impl.cnn_pdtb_classifier_utils_pytorch import load_data_for_mtl, compute_metrices

import torch
torch.manual_seed(57697)
import torch.nn as nn
import torch.nn.functional as F


class LossCompute:
    "A Loss compute and train function."

    def __init__(self, clf_criterion, opt = None):
        self.clf_criterion = clf_criterion
        self.opt = opt

    def __call__(self, Y, clf_logits, only_return_losses = False):
        # Classification loss
        clf_losses = self.clf_criterion(clf_logits, Y)
        # only_return_losses: true for validation and test
        if only_return_losses:
            return clf_losses

        train_loss = clf_losses.sum()
        train_loss.backward()
        if self.opt is not None:
            # Performs a single optimization step
            self.opt.step()
            self.opt.zero_grad()
        return train_loss.item()


class PDTB_Classifier(nn.Module):

    def __init__(self, args):

        super(PDTB_Classifier, self).__init__()

        self.args = args

        # word embedding layer
        self.word_embed = nn.Embedding(args.W.shape[0], args.W.shape[1])
        self.word_embed.weight.data.copy_(torch.from_numpy(args.W))

        if args.emb_static:
            self.word_embed.weight.requires_grad = False

        # pos embedding layer
        self.pos_embed = nn.Embedding(args.P.shape[0], args.P.shape[1])
        self.pos_embed.weight.data.copy_(torch.from_numpy(args.P))
        self.pos_embed.weight.requires_grad = False

        emb_dim = args.W.shape[1] + args.P.shape[1]

        # convolutional layers for arg
        self.conv_layers_arg = nn.ModuleList([nn.Conv2d(1, args.nfmaps_arg, (K, emb_dim), stride = (1, 1)) for K in args.fsz_arg])

        # initialize conv_layers_arg
        for conv_layer in self.conv_layers_arg:
            # nn.init.xavier_uniform_(conv_layer.weight, gain = nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform(conv_layer.weight, gain = nn.init.calculate_gain('relu'))
            conv_layer.bias.data.fill_(0)
            # nn.init.zeros_(conv_layer.bias)

#
#         # dense layers for arg
#         dense_layers_arg = []
#         for i, D in enumerate(args.dsz_arg):
#             if i == 0:
#                 dense_layers_arg.append(nn.Linear(len(args.fsz_arg) * args.nfmaps_arg, D))
#             else:
#                 dense_layers_arg.append(nn.Linear(args.dsz_arg[i - 1], D))
#
#         self.dense_layers_arg = nn.ModuleList(dense_layers_arg)
#
#         # initialize dense_layers_arg
#         for dense_layer in self.dense_layers_arg:
#             nn.init.xavier_uniform_(dense_layer.weight, gain = nn.init.calculate_gain('relu'))
#             dense_layer.bias.data.fill_(0)
#             # nn.init.zeros_(dense_layer.bias)

        # define gate1
        self.dense_arg_cap = nn.Linear(2 * len(args.fsz_arg) * args.nfmaps_arg, args.gate_units_arg)
        # nn.init.xavier_uniform_(self.dense_arg_cap.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform(self.dense_arg_cap.weight, gain = nn.init.calculate_gain('relu'))
        self.dense_arg_cap.bias.data.fill_(0)

        self.dense_arg_gate = nn.Linear(2 * len(args.fsz_arg) * args.nfmaps_arg, args.gate_units_arg)
        # nn.init.xavier_uniform_(self.dense_arg_gate.weight, gain = 1)
        nn.init.xavier_uniform(self.dense_arg_gate.weight, gain = 1)
        self.dense_arg_gate.bias.data.fill_(0)

        # classification layer for imp
        self.clf_layer_imp = nn.Linear(args.gate_units_arg, args.nclasses)
        # nn.init.xavier_uniform_(self.clf_layer_imp.weight, gain = 1)
        nn.init.xavier_uniform(self.clf_layer_imp.weight, gain = 1)
        self.clf_layer_imp.bias.data.fill_(0)

        # classification layer for exp
        self.clf_layer_exp = nn.Linear(args.gate_units_arg, args.nclasses)
        # nn.init.xavier_uniform_(self.clf_layer_exp.weight, gain = 1)
        nn.init.xavier_uniform(self.clf_layer_exp.weight, gain = 1)
        self.clf_layer_exp.bias.data.fill_(0)

    def forward(self, X):

        X_larg, X_lpos, X_rarg, X_rpos, is_imp = X

        h_arg_vecs = []
        for x in [(X_larg, X_lpos), (X_rarg, X_rpos)]:

            x_w, x_p = x

            x_w = self.word_embed(x_w)  # (batch_size, seq_len, dim)
            x_p = self.pos_embed(x_p)

            x_w_p = torch.cat([x_w, x_p], 2)

            x_w_p = x_w_p.unsqueeze(1)  # (batch_size, 1, seq_len, dim)

            x_convs = [F.relu(conv_layer(x_w_p)).squeeze(3) for conv_layer in self.conv_layers_arg]

            # At this point x_convs is [(batch_size, nfmaps_arg, seq_len_new), ...]*len(fsz_arg)
            x_max_pools = [F.max_pool1d(xi, xi.size(2)).squeeze(2) for xi in x_convs]  # [(batch_size, nfmaps_arg), ...]*len(fsz_arg)

            x_max_pool = torch.cat(x_max_pools, 1)

            x_max_pool = nn.Dropout(self.args.dropout_p)(x_max_pool)

            h_arg_vecs.append(x_max_pool)

        h_arg_vec = torch.cat(h_arg_vecs, 1)
        h_arg_cap = F.relu(self.dense_arg_cap(h_arg_vec))
        h_arg_gate = torch.sigmoid(self.dense_arg_gate(h_arg_vec))
        h_clf_in = h_arg_cap * h_arg_gate

        h_out_imp = self.clf_layer_imp(nn.Dropout(self.args.dropout_p)(h_clf_in))
        h_out_exp = self.clf_layer_exp(nn.Dropout(self.args.dropout_p)(h_clf_in))

        is_exp = 1 - is_imp

        # TODO: cast is_imp and is_exp to float tensor
        h_out = is_imp.unsqueeze(1).expand_as(h_out_imp).float() * h_out_imp + is_exp.unsqueeze(1).expand_as(h_out_exp).float() * h_out_exp

        return h_out


def iter_data(datas, batch_size = 200):
    n = int(math.ceil(float(len(datas[0])) / batch_size)) * batch_size
    for i in range(0, n, batch_size):
        if len(datas) == 1:
            yield datas[0][i:i + batch_size]
        else:
            yield [d[i:i + batch_size] for d in datas]


def train(train_set, val_set, args, run_id):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    train_y = [train_set[-1][ii][0] for ii in range(len(train_set[-1]))]
    # At this point, train set labels will be a list where for each example, we have only one label
    train_set = train_set[:-1]
    train_set.append(train_y)

    imp_indices_val = np.where(val_set[-2] == 1)[0]
    exp_indices_val = np.where(val_set[-2] == 0)[0]
    y_val_imp = val_set[-1][imp_indices_val]
    y_val_exp = val_set[-1][exp_indices_val]
    val_set = val_set[:-1]

    # model
    clf = PDTB_Classifier(args)
    # clf = clf.to(device)
    clf = clf.cuda()
    # class_weights = torch.from_numpy(np.asarray(args.class_weights, dtype = 'float32')).to(device)
    class_weights = torch.from_numpy(np.asarray(args.class_weights, dtype = 'float32')).cuda()

    # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
    criterion = nn.CrossEntropyLoss(weight = class_weights)

    print('List of trainable parameters: ')
    for name, param in clf.named_parameters():
    # for name, param in clf.parameters():
        if param.requires_grad:
            print(name)

    model_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, clf.parameters()),
                                 lr = args.lr,
                                 weight_decay = args.l2_weight)

    compute_loss_fct = LossCompute(criterion, model_opt)

    n_epochs = 0

    best_val_perf = None
    if args.conv_metric in ['f1' , 'acc']:
        best_val_perf = 0
    else:
        best_val_perf = np.inf
    patience = args.patience

    for i in range(args.n_epochs):
        n_epochs += 1
        print("running epoch", i)
        # actual training starts here
        start_time = time.time()
        tr_loss = 0
        batch_count = 0
        for _train_set in iter_data(shuffle(*train_set, random_state = np.random),
                                                    batch_size = args.batch_size):
            clf.train()
            # _train_set = [torch.tensor(data, dtype = torch.long).to(device) for data in _train_set]
            _train_set = [torch.LongTensor(data).cuda() for data in _train_set]
            y_tr = _train_set[-1]
            _train_set = _train_set[:-1]
            clf_logits = clf(_train_set)
            tr_loss += compute_loss_fct(y_tr, clf_logits)
            batch_count += 1

        tr_loss /= batch_count
        print('epoch: %i, training time: %.2f secs, train loss: %.4f' % (n_epochs, time.time() - start_time, tr_loss))

        logits = []
        with torch.no_grad():
            clf.eval()
            for _val_set in iter_data(val_set, batch_size = args.batch_size):
                # _val_set = [torch.tensor(data, dtype = torch.long).to(device) for data in _val_set]
                _val_set = [torch.LongTensor(data).cuda() for data in _val_set]
                clf_logits = clf(_val_set)
                logits.append(clf_logits.to("cpu").numpy())

        logits = np.concatenate(logits, 0)
        logits_imp = logits[imp_indices_val]
        logits_exp = logits[exp_indices_val]

        # At present the logits are not probabilities so will only work for multiclass clf
        _, y_true_val_imp, y_pred_val_imp = post_process(-1, logits_imp, y_val_imp, args.multilabel, args.binarize, args.binary_class)

        val_acc_imp, val_macro_f1_imp, val_micro_f1_imp = compute_metrices(y_true_val_imp,
                                                               y_pred_val_imp,
                                                               binary_class = args.binary_class,
                                                               binarize = args.binarize)

        _, y_true_val_exp, y_pred_val_exp = post_process(-1, logits_exp, y_val_exp, args.multilabel, args.binarize, args.binary_class)

        val_acc_exp, val_macro_f1_exp, val_micro_f1_exp = compute_metrices(y_true_val_exp,
                                                               y_pred_val_exp,
                                                               binary_class = args.binary_class,
                                                               binarize = args.binarize)

        print('Imp val acc: %.2f, Imp val micro f1: %.2f, Imp val macro f1: %.2f' % (val_acc_imp * 100.0, val_micro_f1_imp * 100.0, val_macro_f1_imp * 100.0))
        print('Exp val acc: %.2f, Exp val micro f1: %.2f, Exp val macro f1: %.2f' % (val_acc_exp * 100.0, val_micro_f1_exp * 100.0, val_macro_f1_exp * 100.0))

        if args.conv_metric == 'f1':
            val_perf = val_macro_f1_imp
        elif args.conv_metric == 'acc':
            val_perf = val_acc_imp

        if (args.conv_metric in ['f1', 'acc'] and val_perf > best_val_perf):

            best_val_perf = val_perf
            patience = args.patience

            print('Saving the best model . . .')
            path = os.path.join(args.output_dir, 'best_params_{0}'.format(run_id))
            torch.save(clf.state_dict(), path)

        else:
            patience -= 1

        if patience <= 0:
            print ('Early stopping . . .')
            break


def test(test_set, args, run_id):

    imp_indices_test = np.where(test_set[-2] == 1)[0]
    exp_indices_test = np.where(test_set[-2] == 0)[0]
    y_test_imp = test_set[-1][imp_indices_test]
    y_test_exp = test_set[-1][exp_indices_test]
    test_set = test_set[:-1]

    # model
    clf = PDTB_Classifier(args)

    clf.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_params_{0}'.format(run_id))))
    # clf = clf.to(device)
    clf = clf.cuda()

    logits = []
    with torch.no_grad():
        clf.eval()
        for _test_set in iter_data(test_set, batch_size = args.batch_size):
            # _test_set = [torch.tensor(data, dtype = torch.long).to(device) for data in _test_set]
            _test_set = [torch.LongTensor(data).cuda() for data in _test_set]
            clf_logits = clf(_test_set)
            logits.append(clf_logits.to("cpu").numpy())

    logits = np.concatenate(logits, 0)
    logits_imp = logits[imp_indices_test]
    logits_exp = logits[exp_indices_test]

    # At present the logits are not probabilities so will only work for multiclass clf
    _, y_true_test_imp, y_pred_test_imp = post_process(-1, logits_imp, y_test_imp, args.multilabel, args.binarize, args.binary_class)

    test_acc_imp, test_macro_f1_imp, test_micro_f1_imp = compute_metrices(y_true_test_imp,
                                                           y_pred_test_imp,
                                                           binary_class = args.binary_class,
                                                           binarize = args.binarize)

    _, y_true_test_exp, y_pred_test_exp = post_process(-1, logits_exp, y_test_exp, args.multilabel, args.binarize, args.binary_class)

    test_acc_exp, test_macro_f1_exp, test_micro_f1_exp = compute_metrices(y_true_test_exp,
                                                           y_pred_test_exp,
                                                           binary_class = args.binary_class,
                                                           binarize = args.binarize)

    print('Imp test acc: %.2f, Imp test micro f1: %.2f, Imp test macro f1: %.2f' % (test_acc_imp * 100.0, test_micro_f1_imp * 100.0, test_macro_f1_imp * 100.0))
    print('Exp test acc: %.2f, Exp test micro f1: %.2f, Exp test macro f1: %.2f' % (test_acc_exp * 100.0, test_micro_f1_exp * 100.0, test_macro_f1_exp * 100.0))

    pickle.dump([y_true_test_imp, y_pred_test_imp, _ , _, args.class_names], open(os.path.join(args.output_dir, 'best_prediction_imp_' + args.timestamp + '_' + str(run_id) + '.p'), 'wb'))
    pickle.dump([y_true_test_exp, y_pred_test_exp, _ , _, args.class_names], open(os.path.join(args.output_dir, 'best_prediction_exp_' + args.timestamp + '_' + str(run_id) + '.p'), 'wb'))
    print('############################### IMP ###############################')
    print(classification_report(y_true_test_imp, y_pred_test_imp, target_names = args.class_names, labels = range(len(args.class_names)), digits = 4))
    print('############################### EXP ###############################')
    print(classification_report(y_true_test_exp, y_pred_test_exp, target_names = args.class_names, labels = range(len(args.class_names)), digits = 4))

    return test_macro_f1_imp, test_acc_imp, test_macro_f1_exp, test_acc_exp


def main():

    ts = get_time_stamp(args.input_file)

    filter_hs_arg = [2, 3, 4, 5]

    train_data, val_data, test_data, class_dict, examples_per_class = load_data_for_mtl(args.input_file, filter_hs_arg[-1], 2)

    # [X_larg, X_rarg, X_lpos, X_rpos, X_lner, X_rner, X_wp, X_wp_rev, X_wp_pos, X_wp_rev_pos, X_wp_ner, X_wp_rev_ner, is_imp, y]
    X_tr_larg, X_tr_rarg, X_tr_lpos, X_tr_rpos, _, _, _, _, _, _, _, _, is_imp_tr, Y_tr_all = train_data
    X_val_larg, X_val_rarg, X_val_lpos, X_val_rpos, _, _, _, _, _, _, _, _, is_imp_val, Y_val_all = val_data
    X_te_larg, X_te_rarg, X_te_lpos, X_te_rpos, _, _, _, _, _, _, _, _, is_imp_te, Y_te_all = test_data

    train_set = [X_tr_larg, X_tr_lpos, X_tr_rarg, X_tr_rpos, is_imp_tr, Y_tr_all]
    val_set = [X_val_larg, X_val_lpos, X_val_rarg, X_val_rpos, is_imp_val, Y_val_all]
    test_set = [X_te_larg, X_te_lpos, X_te_rarg, X_te_rpos, is_imp_te, Y_te_all]

    args.emb_static = True
    args.reg_emb = False  # should regularize embeddings or not. Obviously when emb_static is True, reg_emb will be false
    args.fsz_arg = filter_hs_arg
    args.nfmaps_arg = 50
    args.dsz_arg = []
    args.gate_units_arg = 300
    args.nclasses = len(class_dict)
    args.binarize = True if len(class_dict) == 2 else False
    args.binary_class = get_binary_class_index(class_dict)
    args.dropout_p = 0.5
    args.l2_weight = 1e-4
    args.batch_size = 200
    args.class_names = get_class_names(class_dict)
    args.n_epochs = 30
    args.patience = 5
    args.timestamp = ts

    print_params(args)

    word2idx, pos2idx, _ = pickle.load(open(args.encoder_file, "rb"))
    args.W = get_W(args.w2v_file, word2idx)
    args.P = np.identity(len(pos2idx))
    args.class_weights = get_class_weights(class_dict, examples_per_class)

    print('Class weights:')
    print(args.class_weights)

    f_scores_imp = []
    accuracies_imp = []
    f_scores_exp = []
    accuracies_exp = []
    runs = range(0, 5)
    for run_id in runs:
        print ('Run: ', run_id)
        train(train_set, val_set, args, run_id)
        f1_imp, acc_imp, f1_exp, acc_exp = test(test_set, args, run_id)
        f_scores_imp.append(f1_imp)
        accuracies_imp.append(acc_imp)
        f_scores_exp.append(f1_exp)
        accuracies_exp.append(acc_exp)

    print('avg f1 (imp): %s' % (str(np.mean(f_scores_imp))))
    print('avg acc (imp): %s' % (str(np.mean(accuracies_imp))))
    print('avg f1 (exp): %s' % (str(np.mean(f_scores_exp))))
    print('avg acc (exp): %s' % (str(np.mean(accuracies_exp))))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = '')

    parser.add_argument('input_file', help = 'pickled input file generated using \'preprocess_pdtb_relations.py\'')
    parser.add_argument('encoder_file', default = None, type = str, help = 'WordEncoder file')
    parser.add_argument('w2v_file', type = str, default = None, help = 'GoogleNews-vectors-negative300.bin')  # GoogleNews-vectors-negative300.bin
    parser.add_argument('output_dir', help = 'directory where you want to save the best model and predictions file')
    parser.add_argument('--conv_metric', default = 'f1', type = str, help = '\'f1\', \'acc\'')
    parser.add_argument('--multilabel', default = True, type = bool, help = 'If True, multilabel evaluation will be done on val and test sets')
    parser.add_argument('--trained_weights_file', default = None, type = str, help = 'file containing the trained model weights')
    parser.add_argument('--opt', default = 'adam', type = str, help = 'opt to use')
    parser.add_argument('--lr', default = 0.0005, type = float, help = 'learning rate to use')

    args = parser.parse_args()

    main()
