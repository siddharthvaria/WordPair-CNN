import argparse
from lasagne.utils import floatX
from lasagne.regularization import regularize_network_params, regularize_layer_params
import time
import os
import pickle
import cPickle
import lasagne
import numpy as np
import theano
import theano.tensor as T
from sklearn.metrics import classification_report

from cnn_pdtb_classifier_utils import get_W, get_loss, post_process, load_model, print_params
from cnn_pdtb_classifier_utils import pad_dataset, shared_dataset, save_model, get_time_stamp
from cnn_pdtb_classifier_utils import get_class_weights, get_class_names, get_binary_class_index
from cnn_pdtb_classifier_utils import load_data_for_mtl, compute_metrices
# theano.config.optimizer = 'fast_compile'
# theano.config.exception_verbosity = 'high'
# theano.config.compute_test_value = 'warn'


class PDTB_Classifier(object):

    def __init__(self, W, P, class_weights, params):

        # self.W = W
        self.params = params

        # define model architecture
        self.index = T.lscalar()
        self.inputs = []
        self.inputs.append([[T.lmatrix('l_arg'), T.lmatrix('l_arg_pos')], [T.lmatrix('r_arg'), T.lmatrix('r_arg_pos')]])
        self.inputs.append([[T.lmatrix('wp'), T.lmatrix('wp_pos')], [T.lmatrix('wp_rev'), T.lmatrix('wp_rev_pos')]])
        self.y = T.ivector('y')
        self.is_imp = T.lvector('is_imp')

        class_weights = T.constant(np.array(class_weights, dtype = theano.config.floatX))

        # the dimension of the input is (batch_size, max_sentence_length)

        # define a dictionary of shared layers of trainable parameters
        self.layers = {}
        self.layers['emb'] = []
        self.layers['pos_emb'] = []
        self.layers['wp'] = {}
        self.layers['arg'] = {}
        self.layers['wp']['conv'] = []
        self.layers['arg']['conv'] = []
        # self.layers['arg']['att'] = []
        self.layers['wp']['dense'] = []
        self.layers['arg']['dense'] = []
        self.layers['penultimate'] = None
        self.layers['output_imp'] = None
        self.layers['output_exp'] = None

        # process word pairs and reverse word pairs
        h_wp_vecs = []
        for x in self.inputs[1]:
            h = self.from_input_to_vec(x,
                                       W,
                                       P,
                                       params['emb_dim'],
                                       params['pos_emb_dim'],
                                       params['filter_hs_wp'],
                                       params['filter_w'],
                                       params['img_h_wp'],
                                       params['n_f_maps_wp'],
                                       params['dense_units_wp'],
                                       params['dropout_p'],
                                       prefix = 'wp',
                                       stride = 2)
            h_wp_vecs.append(h)

        # process left and right arguments
        h_arg_vecs = []
        for x in self.inputs[0]:
            h = self.from_input_to_vec(x,
                                       W,
                                       P,
                                       params['emb_dim'],
                                       params['pos_emb_dim'],
                                       params['filter_hs_arg'],
                                       params['filter_w'],
                                       params['img_h_arg'],
                                       params['n_f_maps_arg'],
                                       params['dense_units_arg'],
                                       params['dropout_p'],
                                       prefix = 'arg',
                                       stride = 1)
            h_arg_vecs.append(h)

        # option 4: gate layer for arg
        h_arg_vec = lasagne.layers.ConcatLayer(h_arg_vecs, axis = 1)
        h_arg_cap = lasagne.layers.DenseLayer(h_arg_vec,
                                              num_units = self.params['gate_units_arg'],
                                              W = lasagne.init.GlorotUniform('relu'),
                                              nonlinearity = lasagne.nonlinearities.rectify,
                                              name = 'h_arg_cap')
        h_arg_gate = lasagne.layers.DenseLayer(h_arg_vec,
                                              num_units = self.params['gate_units_arg'],
                                              W = lasagne.init.GlorotUniform(),
                                              nonlinearity = lasagne.nonlinearities.sigmoid,
                                              name = 'h_arg_gate')
        h_arg = lasagne.layers.ElemwiseMergeLayer([h_arg_cap, h_arg_gate], merge_function = T.mul)

        h_wp_vec = lasagne.layers.ConcatLayer(h_wp_vecs, axis = 1)

        h_final = lasagne.layers.ConcatLayer([h_arg, h_wp_vec], axis = 1)

        h_final_cap = lasagne.layers.DenseLayer(h_final,
                                              num_units = self.params['gate_units_final'],
                                              W = lasagne.init.GlorotUniform('relu'),
                                              nonlinearity = lasagne.nonlinearities.rectify,
                                              name = 'h_final_cap')
        h_final_gate = lasagne.layers.DenseLayer(h_final,
                                              num_units = self.params['gate_units_final'],
                                              W = lasagne.init.GlorotUniform(),
                                              nonlinearity = lasagne.nonlinearities.sigmoid,
                                              name = 'h_final_gate')

        h_clf_in = lasagne.layers.ElemwiseMergeLayer([h_final_cap, h_final_gate], merge_function = T.mul)

        self.layers['penultimate'] = h_clf_in

        h_clf_out_imp = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(h_clf_in, p = params['dropout_p']),
                num_units = self.params['nclasses'],
                W = lasagne.init.GlorotUniform(),
                nonlinearity = lasagne.nonlinearities.softmax,
                name = 'h_clf_out_imp')

        h_clf_out_exp = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(h_clf_in, p = params['dropout_p']),
                num_units = self.params['nclasses'],
                W = lasagne.init.GlorotUniform(),
                nonlinearity = lasagne.nonlinearities.softmax,
                name = 'h_clf_out_exp')

        # output layer
        self.layers['output_imp'] = h_clf_out_imp
        self.layers['output_exp'] = h_clf_out_exp

        if params['emb_static']:
            for l_emb in self.layers['emb']:
                l_emb.params[l_emb.W].remove("trainable")
                l_emb.params[l_emb.W].remove("regularizable")
        else:
            if not params['reg_emb']:
                for l_emb in self.layers['emb']:
                    l_emb.params[l_emb.W].remove("regularizable")

        for l_emb in self.layers['pos_emb']:
            l_emb.params[l_emb.W].remove("trainable")
            l_emb.params[l_emb.W].remove("regularizable")

        # draw_to_file(lasagne.layers.get_all_layers(self.layers['output']), 'network.pdf', output_shape = False)
        # define parameters of the model and update functions using adadelta

        tr_params = lasagne.layers.get_all_params(self.layers['output_imp'], trainable = True)
        tr_params += [h_clf_out_exp.W, h_clf_out_exp.b]

        print 'Trainable parameters: '
        print tr_params

        reg_params = lasagne.layers.get_all_params(self.layers['output_imp'], regularizable = True)
        reg_params += [h_clf_out_exp.W]

        print 'Regularizable parameters: '
        print reg_params

        # handle training loss and predictions
        prediction_imp = lasagne.layers.get_output(self.layers['output_imp'])
        prediction_imp = prediction_imp * self.is_imp.dimshuffle(0, 'x')
        prediction_exp = lasagne.layers.get_output(self.layers['output_exp'])
        prediction_exp = prediction_exp * (1 - self.is_imp.dimshuffle(0, 'x'))
        prediction = prediction_imp + prediction_exp
        _tr_loss = get_loss(params['loss_type'], prediction, self.y)
        self.tr_loss = (lasagne.objectives.aggregate(_tr_loss, weights = class_weights[self.y]) +
        params['l2_weight'] * regularize_network_params(self.layers['output_imp'], lasagne.regularization.l2) +
        params['l2_weight'] * regularize_layer_params(self.layers['output_exp'], lasagne.regularization.l2))

        if params['opt'] == 'adadelta':
            self.grad_updates = lasagne.updates.adadelta(self.tr_loss, tr_params, learning_rate = params['lr'])
        elif params['opt'] == 'adam':
            self.grad_updates = lasagne.updates.adam(self.tr_loss, tr_params, learning_rate = params['lr'])

        # handle test loss and predictions
        self.test_reps = lasagne.layers.get_output(self.layers['penultimate'], deterministic = True)
        test_prediction_imp = lasagne.layers.get_output(self.layers['output_imp'], deterministic = True)
        test_prediction_imp = test_prediction_imp * self.is_imp.dimshuffle(0, 'x')
        test_prediction_exp = lasagne.layers.get_output(self.layers['output_exp'], deterministic = True)
        test_prediction_exp = test_prediction_exp * (1 - self.is_imp.dimshuffle(0, 'x'))
        test_prediction = test_prediction_imp + test_prediction_exp
        _test_loss = get_loss(params['loss_type'], test_prediction, self.y)
        self.test_loss = (lasagne.objectives.aggregate(_test_loss, weights = class_weights[self.y]) +
        params['l2_weight'] * regularize_network_params(self.layers['output_imp'], lasagne.regularization.l2) +
        params['l2_weight'] * regularize_layer_params(self.layers['output_exp'], lasagne.regularization.l2))
        self.y_proba = test_prediction

    def get_filter_pool_sizes(self, filter_heights, filter_w, img_h, stride = 1):
        filter_sizes = []
        pool_sizes = []
        for filter_height in filter_heights:
            # print filter_height
            filter_sizes.append((filter_height, filter_w))
            pool_sizes.append(((img_h - filter_height + stride) // stride, 1))

        return filter_sizes, pool_sizes

    def get_output_conv_pool(self, v, filter_sizes, pool_sizes, nfeature_maps, stride, prefix):

        conv_layers = []
        # att_layers = []
        flatten_layers = []
        if len(self.layers[prefix]['conv']) == 0:
            for ii in xrange(len(filter_sizes)):
                filter_size = filter_sizes[ii]
                pool_size = pool_sizes[ii]
                conv_layer = lasagne.layers.Conv2DLayer(
                    v,
                    num_filters = nfeature_maps,
                    filter_size = filter_size,
                    stride = (stride, 1),
                    nonlinearity = lasagne.nonlinearities.rectify,
                    W = lasagne.init.GlorotUniform('relu'),
                    name = 'conv_{0}_{1}_{2}'.format(prefix, str(len(self.layers[prefix]['conv'])), str(len(conv_layers))))
                conv_layers.append(conv_layer)
                # At this point conv_layer has dim (bs, nfmaps, len-fsz+1, 1)
#                 if prefix == 'arg':
#                     # add attention
#                     conv_layer_tmp = lasagne.layers.FlattenLayer(conv_layer, outdim = 3)
#                     conv_layer_tmp = lasagne.layers.DimshuffleLayer(conv_layer_tmp, (0, 2, 1))
#                     left_att = LeftAttentionLayer(conv_layer_tmp,
#                                                   self.params['img_h_arg'] - filter_size[0] + 1,
#                                                   self.params['n_f_maps_arg'],
#                                                   name = 'larg_att_{0}'.format(str(filter_size[0])))
#                     att_layers.append(left_att)
#                     left_att_tmp = lasagne.layers.DimshuffleLayer(left_att, (0, 2, 1))
#                     left_att_tmp = lasagne.layers.ReshapeLayer(left_att_tmp, ([0], [1], [2], 1))
#                     max_pool_layer = lasagne.layers.MaxPool2DLayer(left_att_tmp, pool_size = pool_size)
#                 else:
                max_pool_layer = lasagne.layers.MaxPool2DLayer(conv_layer, pool_size = pool_size)
                flatten_layer = lasagne.layers.FlattenLayer(max_pool_layer, outdim = 2)
                flatten_layers.append(flatten_layer)
        else:
            for ii in xrange(len(filter_sizes)):
                filter_size = filter_sizes[ii]
                pool_size = pool_sizes[ii]
                conv_layer = lasagne.layers.Conv2DLayer(
                    v,
                    num_filters = nfeature_maps,
                    filter_size = filter_size,
                    stride = (stride, 1),
                    nonlinearity = lasagne.nonlinearities.rectify,
                    W = self.layers[prefix]['conv'][0][ii].W,
                    b = self.layers[prefix]['conv'][0][ii].b,
                    name = 'conv_{0}_{1}_{2}'.format(prefix, str(len(self.layers[prefix]['conv'])), str(len(conv_layers))))
                conv_layers.append(conv_layer)
                # At this point conv_layer has dim (bs, nfmaps, len-fsz+1, 1)
#                 if prefix == 'arg':
#                     # add attention
#                     conv_layer_tmp = lasagne.layers.FlattenLayer(conv_layer, outdim = 3)
#                     conv_layer_tmp = lasagne.layers.DimshuffleLayer(conv_layer_tmp, (0, 2, 1))
#                     right_att = RightAttentionLayer(conv_layer_tmp,
#                                                   self.params['img_h_arg'] - filter_size[0] + 1,
#                                                   self.params['n_f_maps_arg'],
#                                                   W = self.layers['arg']['att'][0][ii].W,
#                                                   name = 'rarg_att_{0}'.format(str(filter_size[0])))
#                     att_layers.append(right_att)
#                     right_att_tmp = lasagne.layers.DimshuffleLayer(right_att, (0, 2, 1))
#                     right_att_tmp = lasagne.layers.ReshapeLayer(right_att_tmp, ([0], [1], [2], 1))
#                     max_pool_layer = lasagne.layers.MaxPool2DLayer(right_att_tmp, pool_size = pool_size)
#                 else:
                max_pool_layer = lasagne.layers.MaxPool2DLayer(conv_layer, pool_size = pool_size)
                flatten_layer = lasagne.layers.FlattenLayer(max_pool_layer, outdim = 2)
                flatten_layers.append(flatten_layer)

        # append the convolutional layers
        self.layers[prefix]['conv'].append(conv_layers)
#         if prefix == 'arg':
#             self.layers['arg']['att'].append(att_layers)

        if len(flatten_layers) > 1:
            conv_layers_op = lasagne.layers.ConcatLayer(flatten_layers, axis = 1)
        else:
            conv_layers_op = flatten_layers[0]

        return conv_layers_op

    def get_output_dense(self, v, dense_units, dropout_p, prefix):

        if len(dense_units) == 0:
            return v

        dense_layers = []
        if len(self.layers[prefix]['dense']) == 0:
            for idx in xrange(len(dense_units)):
                if len(dense_layers) == 0:
                    dense_layer_op = lasagne.layers.DenseLayer(
                        v,
                        num_units = dense_units[idx],
                        nonlinearity = lasagne.nonlinearities.rectify,
                        W = lasagne.init.GlorotUniform('relu'),
                        name = 'dense_{0}_{1}_{2}'.format(prefix, str(len(self.layers[prefix]['dense'])), str(len(dense_layers))))
                else:
                    dense_layer_op = lasagne.layers.DenseLayer(
                        lasagne.layers.dropout(dense_layers[-1], p = dropout_p),
                        num_units = dense_units[idx],
                        nonlinearity = lasagne.nonlinearities.rectify,
                        W = lasagne.init.GlorotUniform('relu'),
                        name = 'dense_{0}_{1}_{2}'.format(prefix, str(len(self.layers[prefix]['dense'])), str(len(dense_layers))))
                dense_layers.append(dense_layer_op)
        else:
            for idx in xrange(len(dense_units)):
                if len(dense_layers) == 0:
                    dense_layer_op = lasagne.layers.DenseLayer(
                        v,
                        num_units = dense_units[idx],
                        nonlinearity = lasagne.nonlinearities.rectify,
                        W = self.layers[prefix]['dense'][0][idx].W,
                        b = self.layers[prefix]['dense'][0][idx].b,
                        name = 'dense_{0}_{1}_{2}'.format(prefix, str(len(self.layers[prefix]['dense'])), str(len(dense_layers))))
                else:
                    dense_layer_op = lasagne.layers.DenseLayer(
                        lasagne.layers.dropout(dense_layers[-1], p = dropout_p),
                        num_units = dense_units[idx],
                        nonlinearity = lasagne.nonlinearities.rectify,
                        W = self.layers[prefix]['dense'][0][idx].W,
                        b = self.layers[prefix]['dense'][0][idx].b,
                        name = 'dense_{0}_{1}_{2}'.format(prefix, str(len(self.layers[prefix]['dense'])), str(len(dense_layers))))
                dense_layers.append(dense_layer_op)

        # append the dense layers
        self.layers[prefix]['dense'].append(dense_layers)

        return dense_layers[-1]

    def from_input_to_vec(self,
                          x,
                          W,
                          P,
                          emb_dim,
                          pos_emb_dim,
                          filter_heights,
                          filter_w,
                          img_h,
                          nfeature_maps,
                          dense_units,
                          dropout_p,
                          prefix,
                          stride = 1):

        # x is the input
        x_w, x_p = x
        l_in_w = lasagne.layers.InputLayer((None, img_h), input_var = x_w)
        l_in_p = lasagne.layers.InputLayer((None, img_h), input_var = x_p)

        if len(self.layers['emb']) == 0:
            l_w_emb = lasagne.layers.EmbeddingLayer(l_in_w,
                                                  input_size = len(W),
                                                  output_size = emb_dim,
                                                  W = floatX(W),
                                                  name = 'emb_' + str(len(self.layers['emb'])))
        else:
            l_w_emb = lasagne.layers.EmbeddingLayer(l_in_w,
                                                  input_size = len(W),
                                                  output_size = emb_dim,
                                                  W = self.layers['emb'][0].W,
                                                  name = 'emb_' + str(len(self.layers['emb'])))

        # append embedding layer
        self.layers['emb'].append(l_w_emb)

        if len(self.layers['pos_emb']) == 0:
            l_pos_emb = lasagne.layers.EmbeddingLayer(l_in_p,
                                                      input_size = len(P),
                                                      output_size = pos_emb_dim,
                                                      W = floatX(P),
                                                      name = 'pos_emb_' + str(len(self.layers['pos_emb'])))
        else:
            l_pos_emb = lasagne.layers.EmbeddingLayer(l_in_p,
                                                      input_size = len(P),
                                                      output_size = pos_emb_dim,
                                                      W = self.layers['pos_emb'][0].W,
                                                      name = 'pos_emb_' + str(len(self.layers['pos_emb'])))

        # append embedding layer
        self.layers['pos_emb'].append(l_pos_emb)

        l_emb = lasagne.layers.ConcatLayer([l_w_emb, l_pos_emb], axis = 2)
        # reshape to add number of channels which is one at this point
        l_emb = lasagne.layers.ReshapeLayer(l_emb, ([0], 1, [1], [2]))

        filter_sizes, pool_sizes = self.get_filter_pool_sizes(filter_heights, filter_w, img_h, stride)

        conv_layers_op = self.get_output_conv_pool(l_emb, filter_sizes, pool_sizes, nfeature_maps, stride, prefix = prefix)

        conv_layers_op = lasagne.layers.dropout(conv_layers_op, p = dropout_p)

        # At the end of convolution, we are going to obtain a single value for each feature map and filter height combination (due to max pooling)
        dense_layers_op = self.get_output_dense(conv_layers_op, dense_units, dropout_p, prefix = prefix)

        return dense_layers_op

    def train(self, train_set, dev_set, test_set, run_id):

        # used only in 2-way tasks
        pos_label = self.params['binary_class']
        # used only in 2-way tasks
        binarize = self.params['binarize']  # True or False

        multilabel = self.params['multilabel']  # True or False

        batch_size = self.params['batch_size']

        train_set, train_sz = pad_dataset(train_set, batch_size)
        dev_set, dev_sz = pad_dataset(dev_set, batch_size)
        test_set, test_sz = pad_dataset(test_set, batch_size)

        n_train_batches = len(train_set[0]) / batch_size
        n_test_batches = len(test_set[0]) / batch_size
        n_dev_batches = len(dev_set[0]) / batch_size

        train_y = [train_set[-1][ii][0] for ii in xrange(len(train_set[-1]))]
        # At this point, train set labels will be a list where for each example, we have only one label
        train_set = train_set[:-1]
        train_set.append(train_y)
        train_set_s = shared_dataset(train_set)

        _dev_y = dev_set[-1][:dev_sz]
        _is_imp_dev = dev_set[-2][:dev_sz]
        imp_indices_dev = np.where(_is_imp_dev == 1)[0]
        exp_indices_dev = np.where(_is_imp_dev == 0)[0]
        dev_set = dev_set[:-1]
        dev_set.append(None)
        dev_set_s = shared_dataset(dev_set)

        _test_y = test_set[-1][:test_sz]
        _is_imp_test = test_set[-2][:test_sz]
        imp_indices_test = np.where(_is_imp_test == 1)[0]
        exp_indices_test = np.where(_is_imp_test == 0)[0]
        test_set = test_set[:-1]
        test_set.append(None)
        test_set_s = shared_dataset(test_set)

        # [[l_arg, l_arg_pos], [r_arg, r_arg_pos]]
        # [[wp, wp_pos], [wp_rev, wp_rev_pos]]
        val_fn1 = theano.function(inputs = [self.index], outputs = [self.y_proba, self.test_reps],
             givens = {
                 # l_arg
                self.inputs[0][0][0]: dev_set_s[0][self.index * batch_size: (self.index + 1) * batch_size],
                # l_arg_pos
                self.inputs[0][0][1]: dev_set_s[1][self.index * batch_size: (self.index + 1) * batch_size],
                # r_arg
                self.inputs[0][1][0]: dev_set_s[2][self.index * batch_size: (self.index + 1) * batch_size],
                # r_arg_pos
                self.inputs[0][1][1]: dev_set_s[3][self.index * batch_size: (self.index + 1) * batch_size],
                # wp
                self.inputs[1][0][0]: dev_set_s[4][self.index * batch_size: (self.index + 1) * batch_size],
                # wp_pos
                self.inputs[1][0][1]: dev_set_s[5][self.index * batch_size: (self.index + 1) * batch_size],
                # wp_rev
                self.inputs[1][1][0]: dev_set_s[6][self.index * batch_size: (self.index + 1) * batch_size],
                # wp_rev_pos
                self.inputs[1][1][1]: dev_set_s[7][self.index * batch_size: (self.index + 1) * batch_size],
                # is_imp
                self.is_imp: dev_set_s[8][self.index * batch_size: (self.index + 1) * batch_size]})

        test_fn1 = theano.function([self.index], outputs = [self.y_proba, self.test_reps],
             givens = {
                 # l_arg
                self.inputs[0][0][0]: test_set_s[0][self.index * batch_size: (self.index + 1) * batch_size],
                # l_arg_pos
                self.inputs[0][0][1]: test_set_s[1][self.index * batch_size: (self.index + 1) * batch_size],
                # r_arg
                self.inputs[0][1][0]: test_set_s[2][self.index * batch_size: (self.index + 1) * batch_size],
                # r_arg_pos
                self.inputs[0][1][1]: test_set_s[3][self.index * batch_size: (self.index + 1) * batch_size],
                # wp
                self.inputs[1][0][0]: test_set_s[4][self.index * batch_size: (self.index + 1) * batch_size],
                # wp_pos
                self.inputs[1][0][1]: test_set_s[5][self.index * batch_size: (self.index + 1) * batch_size],
                # wp_rev
                self.inputs[1][1][0]: test_set_s[6][self.index * batch_size: (self.index + 1) * batch_size],
                # wp_rev_pos
                self.inputs[1][1][1]: test_set_s[7][self.index * batch_size: (self.index + 1) * batch_size],
                # is_imp
                self.is_imp: test_set_s[8][self.index * batch_size: (self.index + 1) * batch_size]})

        train_fn = theano.function([self.index], outputs = self.tr_loss, updates = self.grad_updates,
              givens = {
                 # l_arg
                self.inputs[0][0][0]: train_set_s[0][self.index * batch_size: (self.index + 1) * batch_size],
                # l_arg_pos
                self.inputs[0][0][1]: train_set_s[1][self.index * batch_size: (self.index + 1) * batch_size],
                # r_arg
                self.inputs[0][1][0]: train_set_s[2][self.index * batch_size: (self.index + 1) * batch_size],
                # r_arg_pos
                self.inputs[0][1][1]: train_set_s[3][self.index * batch_size: (self.index + 1) * batch_size],
                # wp
                self.inputs[1][0][0]: train_set_s[4][self.index * batch_size: (self.index + 1) * batch_size],
                # wp_pos
                self.inputs[1][0][1]: train_set_s[5][self.index * batch_size: (self.index + 1) * batch_size],
                # wp_rev
                self.inputs[1][1][0]: train_set_s[6][self.index * batch_size: (self.index + 1) * batch_size],
                # wp_rev_pos
                self.inputs[1][1][1]: train_set_s[7][self.index * batch_size: (self.index + 1) * batch_size],
                # is_imp
                self.is_imp: train_set_s[8][self.index * batch_size: (self.index + 1) * batch_size],
                self.y: train_set_s[9][self.index * batch_size:(self.index + 1) * batch_size]})

        print 'Training for ' + str(self.params['n_epochs']) + ' epoch(s) . . .'

        epoch = 0
        if self.params['conv_metric'] in ['f1' , 'acc']:
            best_val_perf = 0
        else:
            best_val_perf = np.inf

        patience = self.params['patience']
        bbt = 0.5

        while (epoch < self.params['n_epochs']):

            start_time = time.time()
            epoch = epoch + 1

            train_batches_seqs = range(n_train_batches)

            if self.params['shuffle_batch']:
                train_batches_seqs = np.random.permutation(range(n_train_batches))

            tr_loss = 0
            for m_idx in train_batches_seqs:
                tr_loss += train_fn(m_idx)

            tr_loss /= n_train_batches
            print('epoch: %i, training time: %.2f secs, train loss: %.4f' % (epoch, time.time() - start_time, tr_loss))

            dev_y_proba = []
            for m_idx in xrange(n_dev_batches):
                y_proba, _ = val_fn1(m_idx)
                dev_y_proba.extend(y_proba)

            # get rid of padded examples
            dev_y_proba = dev_y_proba[:dev_sz]
            dev_y_proba = np.asarray(dev_y_proba)
            dev_y_proba_imp = dev_y_proba[imp_indices_dev]
            dev_y_proba_exp = dev_y_proba[exp_indices_dev]

            # separate out imp and exp
            _dev_y_imp = _dev_y[imp_indices_dev]
            _dev_y_exp = _dev_y[exp_indices_dev]

            bti, y_true_dev_imp, y_pred_dev_imp = post_process(-1, dev_y_proba_imp, _dev_y_imp, multilabel, binarize, pos_label)
            # print('bti: %.4f' % (bti))
            val_acc_imp, val_macro_f1_imp, val_micro_f1_imp = compute_metrices(y_true_dev_imp,
                                                                   y_pred_dev_imp,
                                                                   binary_class = pos_label,
                                                                   binarize = binarize)

            bte, y_true_dev_exp, y_pred_dev_exp = post_process(-1, dev_y_proba_exp, _dev_y_exp, multilabel, binarize, pos_label)
            # print('bte: %.4f' % (bte))
            val_acc_exp, val_macro_f1_exp, val_micro_f1_exp = compute_metrices(y_true_dev_exp,
                                                                   y_pred_dev_exp,
                                                                   binary_class = pos_label,
                                                                   binarize = binarize)

            print('Imp val acc: %.2f, Imp val micro f1: %.2f, Imp val macro f1: %.2f' % (val_acc_imp * 100.0, val_micro_f1_imp * 100.0, val_macro_f1_imp * 100.0))
            print('Exp val acc: %.2f, Exp val micro f1: %.2f, Exp val macro f1: %.2f' % (val_acc_exp * 100.0, val_micro_f1_exp * 100.0, val_macro_f1_exp * 100.0))

            if self.params['conv_metric'] == 'f1':
                val_perf = val_macro_f1_imp
            elif self.params['conv_metric'] == 'acc':
                val_perf = val_acc_imp

            if (self.params['conv_metric'] in ['f1', 'acc'] and val_perf > best_val_perf):

                patience = self.params['patience']  # renew patience
                bbt = bti

                best_val_perf = val_perf
                test_y_proba = []
                test_reps = []

                for m_idx in xrange(n_test_batches):
                    v1, v2 = test_fn1(m_idx)
                    test_y_proba.extend(v1)
                    test_reps.extend(v2)

                # get rid of padded examples
                test_y_proba = test_y_proba[:test_sz]
                test_y_proba = np.asarray(test_y_proba)
                test_y_proba_imp = test_y_proba[imp_indices_test]
                test_y_proba_exp = test_y_proba[exp_indices_test]

                # separate out imp and exp
                _test_y_imp = _test_y[imp_indices_test]
                _test_y_exp = _test_y[exp_indices_test]
                test_reps = test_reps[:test_sz]
                test_reps = np.asarray(test_reps)
                test_reps_imp = test_reps[imp_indices_test]

                _, y_true_test_imp, y_pred_test_imp = post_process(bti, test_y_proba_imp, _test_y_imp, multilabel, binarize, pos_label)
                test_acc_imp, test_macro_f1_imp, test_micro_f1_imp = compute_metrices(y_true_test_imp,
                                                                       y_pred_test_imp,
                                                                       binary_class = pos_label,
                                                                       binarize = binarize)

                _, y_true_test_exp, y_pred_test_exp = post_process(bte, test_y_proba_exp, _test_y_exp, multilabel, binarize, pos_label)
                test_acc_exp, test_macro_f1_exp, test_micro_f1_exp = compute_metrices(y_true_test_exp,
                                                                       y_pred_test_exp,
                                                                       binary_class = pos_label,
                                                                       binarize = binarize)

                print('Imp test acc: %.2f, Imp test micro f1: %.2f, Imp test macro f1: %.2f' % (test_acc_imp * 100.0, test_micro_f1_imp * 100.0, test_macro_f1_imp * 100.0))
                print('Exp test acc: %.2f, Exp test micro f1: %.2f, Exp test macro f1: %.2f' % (test_acc_exp * 100.0, test_micro_f1_exp * 100.0, test_macro_f1_exp * 100.0))

                print 'Saving the best model . . .'
                best_model = lasagne.layers.get_all_param_values(self.layers['output_imp'])
                best_model += [self.layers['output_exp'].W.get_value(), self.layers['output_exp'].b.get_value()]

                save_model(os.path.join(self.params['output_dir'], 'model_' + self.params['timestamp'] + '_' + str(run_id) + '.p'), best_model)

            else:
                patience = patience - 1

            if patience <= 0:
                print 'Early stopping . . .'
                break

        cPickle.dump([y_true_test_imp, y_pred_test_imp, test_reps_imp , _, self.params['class_names']], open(os.path.join(self.params['output_dir'], 'best_prediction_imp_' + self.params['timestamp'] + '_' + str(run_id) + '.p'), 'wb'))
        cPickle.dump([y_true_test_exp, y_pred_test_exp, _ , _, self.params['class_names']], open(os.path.join(self.params['output_dir'], 'best_prediction_exp_' + self.params['timestamp'] + '_' + str(run_id) + '.p'), 'wb'))
        # print('bbt: %.4f' % (bbt))
        print('############################### IMP ###############################')
        print(classification_report(y_true_test_imp, y_pred_test_imp, target_names = self.params['class_names'], digits = 4))
        print('############################### EXP ###############################')
        print(classification_report(y_true_test_exp, y_pred_test_exp, target_names = self.params['class_names'], digits = 4))

        return test_macro_f1_imp, test_acc_imp, test_macro_f1_exp, test_acc_exp


def main():

    ts = get_time_stamp(args['input_file'])

    emb_static = True
    reg_emb = False  # should regularize embeddings or not. Obviously when emb_static is True, reg_emb will be false

    filter_hs_arg = [2, 3, 4, 5]
    filter_hs_wp = [2, 4, 6, 8]

    # train_data, val_data, test_data, class_dict, examples_per_class = load_data_for_mtl(args['input_file'], filter_hs_arg[-1], filter_hs_wp[-1])
    train_data, val_data, test_data, class_dict, examples_per_class = load_data_for_mtl(args['input_file'], filter_hs_arg[-1], filter_hs_wp[-1])
    word2idx, pos2idx, _ = pickle.load(open(args['encoder_file'], "rb"))

    W = get_W(args['w2v_file'], word2idx)
    P = np.identity(len(pos2idx))

    # [X_larg, X_rarg, X_lpos, X_rpos, X_lner, X_rner, X_wp, X_wp_rev, X_wp_pos, X_wp_rev_pos, X_wp_ner, X_wp_rev_ner, is_imp, y]
    X_tr_larg, X_tr_rarg, X_tr_lpos, X_tr_rpos, _, _, X_tr_wp, X_tr_wp_rev, X_tr_wp_pos, X_tr_wp_rev_pos, _, _, is_imp_tr, Y_tr_all = train_data
    X_val_larg, X_val_rarg, X_val_lpos, X_val_rpos, _, _, X_val_wp, X_val_wp_rev, X_val_wp_pos, X_val_wp_rev_pos, _, _, is_imp_val, Y_val_all = val_data
    X_te_larg, X_te_rarg, X_te_lpos, X_te_rpos, _, _, X_te_wp, X_te_wp_rev, X_te_wp_pos, X_te_wp_rev_pos, _, _, is_imp_te, Y_te_all = test_data

    train_set = [X_tr_larg, X_tr_lpos, X_tr_rarg, X_tr_rpos, X_tr_wp, X_tr_wp_pos, X_tr_wp_rev, X_tr_wp_rev_pos, is_imp_tr, Y_tr_all]
    dev_set = [X_val_larg, X_val_lpos, X_val_rarg, X_val_rpos, X_val_wp, X_val_wp_pos, X_val_wp_rev, X_val_wp_rev_pos, is_imp_val, Y_val_all]
    test_set = [X_te_larg, X_te_lpos, X_te_rarg, X_te_rpos, X_te_wp, X_te_wp_pos, X_te_wp_rev, X_te_wp_rev_pos, is_imp_te, Y_te_all]

    class_weights = get_class_weights(class_dict, examples_per_class)

    class_names = get_class_names(class_dict)

    params = {
      'emb_dim': 300,  # word emb dim
      'pos_emb_dim': len(P),
      'filter_hs_arg': filter_hs_arg,
      'filter_hs_wp':filter_hs_wp,
      'filter_w': 300 + len(P),  # this will change if we augment POS embeddings
      'n_f_maps_arg': 50,
      'n_f_maps_wp': 100,
      'dense_units_arg': [],
      'dense_units_wp': [],
      'gate_units_arg': 300,
      'gate_units_final': 300,
      'nclasses': len(class_dict),
      'binarize': True if len(class_dict) == 2 else False,
      'binary_class': get_binary_class_index(class_dict),
      'dropout_p': 0.5,
      'shuffle_batch': True,
      'n_epochs': 30,
      'batch_size': 200,
      'loss_type': 'categorical_crossentropy',
      'l2_weight': 1e-4,
      'emb_static': emb_static,
      'reg_emb': reg_emb,
      'use_batch_norm': False,
      'timestamp': ts,
      'class_names': class_names,
      'output_dir': args['output_dir'],
      'trained_weights_file': args['trained_weights_file'],
      'img_h_arg': len(train_set[0][0]),
      'img_h_wp': len(train_set[4][0]),
      'multilabel': args['multilabel'],
      'conv_metric': args['conv_metric'],
      'opt': args['opt'],
      'lr': args['lr'],
      'patience': 5
      }

    print_params(params)

    f_scores_imp = []
    accuracies_imp = []
    f_scores_exp = []
    accuracies_exp = []
    runs = range(0, 10)
    for run in runs:
        print 'Run: ', run
        clf = PDTB_Classifier(W, P, class_weights, params)
        f1_imp, acc_imp, f1_exp, acc_exp = clf.train(train_set, dev_set, test_set, run)
        f_scores_imp.append(f1_imp)
        accuracies_imp.append(acc_imp)
        f_scores_exp.append(f1_exp)
        accuracies_exp.append(acc_exp)

    print 'avg f1 (imp): %s' % (str(np.mean(f_scores_imp)))
    print 'avg acc (imp): %s' % (str(np.mean(accuracies_imp)))
    print 'avg f1 (exp): %s' % (str(np.mean(f_scores_exp)))
    print 'avg acc (exp): %s' % (str(np.mean(accuracies_exp)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = '')

    parser.add_argument('input_file', help = 'pickled input file generated using \'preprocess_pdtb_relations.py\'')
    parser.add_argument('encoder_file', default = None, type = str, help = 'WordEncoder file')
    parser.add_argument('w2v_file', type = str, default = None, help = 'GoogleNews-vectors-negative300.bin')  # GoogleNews-vectors-negative300.bin
    parser.add_argument('output_dir', help = 'directory where you want to save the best model and predictions file')
    parser.add_argument('--conv_metric', default = 'f1', type = str, help = '\'f1\', \'acc\'')
    parser.add_argument('--multilabel', default = True, type = bool, help = 'If True, multilabel evaluation will be done on dev and test sets')
    parser.add_argument('--trained_weights_file', default = None, type = str, help = 'file containing the trained model weights')
    parser.add_argument('--opt', default = 'adam', type = str, help = 'opt to use')
    parser.add_argument('--lr', default = 0.0005, type = float, help = 'learning rate to use')

    args = vars(parser.parse_args())

    main()
