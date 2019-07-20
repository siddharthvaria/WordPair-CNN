import argparse
import os

import pickle
from common.text_utils import add_connective
# from text_preprocessing import annotate
# from text_preprocessing import clean_text

RELATIONS_SEPARATOR = u"________________________________________________________"
TEXT_COMMENT = u"#### Text ####"
FEATURES_COMMENT = u"#### Features ####"
END_TEXT_COMMENT = u"##############"
ARG1_LINE = u"____Arg1____"
ARG2_LINE = u"____Arg2____"
SUP1_LINE = u"____Sup1____"
SUP2_LINE = u"____Sup2____"

verbose = False

# def remove_non_ascii(text):
#     # remove non-ascii characters
#     text = re.sub(r'[^\x00-\x7F]+', ' ', text)
#     return text.strip()

# def annotate(argText):
#         argText = remove_non_ascii(argText)
#         if len(argText) == 0:
#             return ''
#         _tmp_op = nlp_server.annotate(argText, properties = {'annotators': 'tokenize, ssplit', 'outputFormat': 'json'})
#         if verbose:
#             print '###############################################'
#             print 'argText: '
#             print argText
#             print '###############################################'
#             print '_tmp_op: '
#             print _tmp_op
#             print '###############################################'
#             print '_tmp_op json: '
#             print json.dumps(_tmp_op, sort_keys = True, indent = 4)
#             print '###############################################'
#         if not 'sentences' in _tmp_op:
#             print 'wtf?!'
#         argText_ann = []
#         for s in _tmp_op['sentences']:
#             csentence = []
#             for t in s['tokens']:
#                 csentence.append(t['word'])
#             argText_ann.append(' '.join(csentence))
#
#         return ''.join(argText_ann)


def parseExplicitRelation(relationLines):

    relation = {}
    relation[u'type'] = u'Explicit'

    index = 1

    # the first line is the span list
    text_span1 = {u'from': [], u'to': [], u'text': None}
    for span in relationLines[index].split(u';'):
        spanList = span.split(u'..')
        text_span1[u'from'].append(int(spanList[0]))
        text_span1[u'to'].append(int(spanList[1]))

    index += 1
    # the second line is the gorn address list
    # TODO: add gorn address
    index += 1

    if not relationLines[index] == TEXT_COMMENT:
        raise ValueError(u'Bad Relation format!')

    index += 1

    text = u''
    while not relationLines[index] == END_TEXT_COMMENT:
        text += (relationLines[index] + u'\n')
        index += 1

    # text_span1[u'text'] = annotate(clean_text(text[:-1]))
    text_span1[u'text'] = text[:-1]
    relation[u'text'] = text_span1

    index += 1

    # next should be the features
    if not relationLines[index] == FEATURES_COMMENT:
        raise ValueError('Bad Relation format!')

    index += 1
    # TODO:
    index += 1  # skip the features for now so just increment index

    # TODO: parse SupportingText
    if u'..' in relationLines[index]:
        while not relationLines[index] == END_TEXT_COMMENT:
            index += 1
        index += 1

    # there will be relation/marker
    if u', ' in relationLines[index]:
        markerAndRelation = relationLines[index].split(u", ")
        # print 'number of classes: ', (len(markerAndRelation) - 1)
        relation[u'marker'] = markerAndRelation[0]
        rel_classes = []
        for rel_class in markerAndRelation[1:]:
            rel_classes.append(rel_class.replace(u" ", u"_"))
        relation[u'class'] = rel_classes
        index += 1

    # now the args. sup1 comes first if it exists, then arg1 and arg2, then sup2 if it exists
    arg1, arg2 = parseArgs(relationLines[index:])

    relation[u'arg1'] = arg1
    arg2['text_span']['text_wc'] = add_connective(arg2['text_span']['text'], relation[u'marker'])
    relation[u'arg2'] = arg2
    return relation


def parseImplicitRelation(relationLines):

    relation = {}
    relation[u'type'] = u'Implicit'
    index = 1
    # first is the inference site
    relation[u'stringPosition'] = int(relationLines[index])
    index += 1
    relation[u'sentenceNumber'] = int(relationLines[index])
    index += 1
    # next should be the features
    if not relationLines[index] == FEATURES_COMMENT:
        raise ValueError('Bad Relation format!')
    index += 1
    # TODO:
    index += 1  # skip the features for now so just increment index

    # TODO: parse SupportingText
    if u'..' in relationLines[index]:
        while not relationLines[index] == END_TEXT_COMMENT:
            index += 1
        index += 1

    # there may be relation/marker
    if u', ' in relationLines[index]:
        markerAndRelation = relationLines[index].split(u", ")
        # print 'number of classes: ', (len(markerAndRelation) - 1)
        relation[u'marker'] = markerAndRelation[0]
        rel_classes = []
        for rel_class in markerAndRelation[1:]:
            rel_classes.append(rel_class.replace(u" ", u"_"))
        relation[u'class'] = rel_classes
        index += 1

    # now the args. sup1 comes first if it exists, then arg1 and arg2, then sup2 if it exists
    arg1, arg2 = parseArgs(relationLines[index:])

    relation[u'arg1'] = arg1
    arg2['text_span']['text_wc'] = arg2['text_span']['text']
    relation[u'arg2'] = arg2
    return relation


def findIndex(lines, s):
    index = 0
    for line in lines:
        if line == s:
            return index
        index += 1
    return -2


def parseArgs(lines):

    # figure out indices
    arg1Start = findIndex(lines, ARG1_LINE) + 1
    arg2Start = findIndex(lines, ARG2_LINE) + 1
    sup1Start = findIndex(lines, SUP1_LINE) + 1
    sup2Start = findIndex(lines, SUP2_LINE) + 1
    if arg1Start == -1 or arg2Start == -1:
        raise ValueError('Bad Relation format!')

    # sup1, arg1, arg2, sup2
    arg1End = arg2Start
    arg2End = len(lines) if sup2Start == -1 else sup2Start
    sup1End = arg1Start
    sup2End = len(lines)

    arg1 = parseArg(lines, arg1Start, arg1End, sup1Start, sup1End)
    arg2 = parseArg(lines, arg2Start, arg2End, sup2Start, sup2End)

    return arg1, arg2


def parseArg(lines, argStart, argEnd, supStart, supEnd):

    arg = {u'text_span': None, u'supplement': None}
    index = argStart
    # the first line is the span list
    text_span1 = {u'from': [], u'to': [], u'text': None}
    for span in lines[index].split(u';'):
        spanList = span.split(u'..')
        text_span1[u'from'].append(int(spanList[0]))
        text_span1[u'to'].append(int(spanList[1]))

    index += 1
    index += 1  # the second line is the gorn address list. Ignore it for now

    if not lines[index] == TEXT_COMMENT:
        raise ValueError('Bad Relation format!')

    index += 1

    text = u''
    while not lines[index] == END_TEXT_COMMENT:
        text += lines[index] + u'\n'
        index += 1

    if verbose:
        print (u'Text before annotation: ', text[:-1])

    # text_span1[u'text'] = annotate(clean_text(text[:-1]))
    text_span1[u'text'] = text[:-1]

    if verbose:
        print (u'Text after annotation: ', text_span1[u'text'])

    arg[u'text_span'] = text_span1

    index += 1
    # TODO: process features inside the argument

    if supStart > -1:
        index = supStart
        text_span2 = {u'from': [], u'to': [], u'text': None}
        for span in lines[index].split(u';'):
            spanList = span.split(u'..')
            text_span2[u'from'].append(int(spanList[0]))
            text_span2[u'to'].append(int(spanList[1]))

        index += 1
        index += 1  # the second line is the gorn address list. Ignore it for now

        if not lines[index] == TEXT_COMMENT:
            raise ValueError('Bad Relation format!')

        index += 1

        text = u''
        while not lines[index] == END_TEXT_COMMENT:
            text += lines[index] + u'\n'
            index += 1

        # text_span2[u'text'] = annotate(clean_text(text[:-1]))
        text_span2[u'text'] = text[:-1]

        arg[u'supplement'] = text_span2

    return arg


def parseAltLexRelation(relationLines):

    relation = {}
    relation[u'type'] = u'AltLex'

    index = 1

    # the first line is the span list
    text_span1 = {u'from': [], u'to': [], u'text': None}
    for span in relationLines[index].split(u';'):
        spanList = span.split(u'..')
        text_span1[u'from'].append(int(spanList[0]))
        text_span1[u'to'].append(int(spanList[1]))

    index += 1
    # the second line is the gorn address list
    # TODO: add gorn address
    index += 1

    if not relationLines[index] == TEXT_COMMENT:
        raise ValueError('Bad Relation format!')

    index += 1

    text = u''
    while not relationLines[index] == END_TEXT_COMMENT:
        text += (relationLines[index] + u'\n')
        index += 1

    # text_span1[u'text'] = annotate(clean_text(text[:-1]))
    text_span1[u'text'] = text[:-1]
    relation[u'text'] = text_span1

    index += 1

    # next should be the features
    if not relationLines[index] == FEATURES_COMMENT:
        raise ValueError('Bad Relation format!')

    index += 1
    # TODO:
    index += 1  # skip the features for now so just increment index

    # TODO: parse SecondText
    if u'..' in relationLines[index]:
        while not relationLines[index] == END_TEXT_COMMENT:
            index += 1
        index += 1

    # there will be relation/marker
    markerAndRelation = relationLines[index].split(u", ")
    if len(markerAndRelation) == 2:
        if markerAndRelation[0] == markerAndRelation[0].lower():
            relation[u'marker'] = markerAndRelation[0]
            relation[u'class'] = [markerAndRelation[1].replace(u" ", u"_")]
        else:
            relation[u'marker'] = u'N/A'
            relation[u'class'] = [markerAndRelation[0].replace(u" ", u"_")]
            relation[u'alt_class'] = [markerAndRelation[1].replace(u" ", u"_")]
    elif len(markerAndRelation) == 1:
        relation[u'marker'] = u'N/A'
        relation[u'class'] = [markerAndRelation[0].replace(u" ", u"_")]
    else:
        raise ValueError('Bad Relation format!')

    index += 1

    # now the args. sup1 comes first if it exists, then arg1 and arg2, then sup2 if it exists
    arg1, arg2 = parseArgs(relationLines[index:])

    relation[u'arg1'] = arg1
    relation[u'arg2'] = arg2

    return relation


def parseEntRelRelation(relationLines):

    relation = {}
    relation[u'type'] = u'EntRel'
    index = 1
    # first is the inference site
    relation[u'stringPosition'] = int(relationLines[index])
    index += 1
    relation[u'sentenceNumber'] = int(relationLines[index])
    index += 1
    # now the args. sup1 comes first if it exists, then arg1 and arg2, then sup2 if it exists
    arg1, arg2 = parseArgs(relationLines[index:])

    relation[u'arg1'] = arg1
    relation[u'arg2'] = arg2

    relation[u'marker'] = u'N/A'
    relation[u'class'] = [u'EntRel']

    return relation


def parseNoRelRelation(relationLines):

    relation = {}
    relation[u'type'] = u'NoRel'
    index = 1
    # first is the inference site
    relation[u'stringPosition'] = int(relationLines[index])
    index += 1
    relation[u'sentenceNumber'] = int(relationLines[index])
    index += 1

    # now the args. sup1 comes first if it exists, then arg1 and arg2, then sup2 if it exists
    arg1, arg2 = parseArgs(relationLines[index:])

    relation[u'arg1'] = arg1
    relation[u'arg2'] = arg2

    relation[u'marker'] = u'N/A'
    relation[u'class'] = [u'NoRel']

    return relation


def parseRelation(relationLines):
    if not relationLines[0].startswith(u"____"):
        raise ValueError('Bad Relation format!')

    _type = relationLines[0].replace(u"____", u"")

    if _type == u'Explicit':
        return parseExplicitRelation(relationLines)

    if _type == u'Implicit':
        return parseImplicitRelation(relationLines)

    if _type == u'AltLex':
        return parseAltLexRelation(relationLines)

    if _type == u'EntRel':
        return parseEntRelRelation(relationLines)

    if _type == u'NoRel':
        return parseNoRelRelation(relationLines)


def loadDocument(filePath, fileNumber, sectionNumber):
    doc = {}
    doc[u'relations'] = []
    doc[u'sectionNumber'] = sectionNumber
    doc[u'fileNumber'] = fileNumber
    with open(filePath, 'r', encoding = 'utf-8', errors = 'replace') as fh:
        if verbose:
            print (u'Processing ', filePath)
        relationLines = []
        for line in fh:
            line = line.strip()
            if line == RELATIONS_SEPARATOR:
                if len(relationLines) > 0:
                    # end of relation in file
                    relation = parseRelation(relationLines)
                    print(u'-------------------------------')
                    print(u'arg1: ', relation[u'arg1'][u'text_span'][u'text'])
                    print(u'arg2: ', relation[u'arg2'][u'text_span'][u'text'])
                    print(u'classes: ', relation[u'class'])
                    print(u'-------------------------------')
                    if relation == None:
                        relationLines = []
                        continue
                    doc[u'relations'].append(relation)
                    relationLines = []
                else:
                    # beginning of a new relation in file
                    relationLines = []
            else:
                relationLines.append(line)
    return doc


def read_sections(basepath, sections_list):
    wsj_files = []
    for section in sections_list:
        onlyfiles = [f for f in os.listdir(os.path.join(basepath, section)) if os.path.isfile(os.path.join(os.path.join(basepath, section), f)) and f.startswith('wsj_')]
        # print onlyfiles
        wsj_files += onlyfiles

    print (u'Number of files to be processed: ', len(wsj_files))

    dataset = []
    wsj_files.sort()
    for fileName in wsj_files:
        fileNumber = fileName[fileName.index('_') + 1:fileName.index('.')]
        sectionNumber = fileNumber[:2]
        dataset.append(loadDocument(os.path.join(os.path.join(basepath, sectionNumber), fileName), fileNumber, sectionNumber))

    return dataset


def read_pdtb(pdtb_data_dir_path, output_dir, task_type):

    if task_type == u'4_way' or task_type == u'11_way':
        # This setting is called PDTB-Ji
        # PDTB-Ji uses sections 2-20, 0-1 and 21-22 as train, dev and test set resp.
        train_sections = [u"%02d" % (i,) for i in range(2, 21)]
        # dev_sections = ["%02d" % (i,) for i in xrange(0, 2)] + ["%02d" % (i,) for i in xrange(23, 25)]
        dev_sections = [u"%02d" % (i,) for i in range(0, 2)]
        test_sections = [u"%02d" % (i,) for i in range(21, 23)]
        sections = {u'train': train_sections, u'test': test_sections, u'dev': dev_sections}
    elif task_type == u'15_way':
        # This setting is called PDTB-Lin
        # PDTB-Lin uses sections 2-21, 22 and 23 as train, dev and test set resp.
        train_sections = [u"%02d" % (i,) for i in range(2, 22)]
        dev_sections = [u"%02d" % (i,) for i in range(22, 23)]
        test_sections = [u"%02d" % (i,) for i in range(23, 24)]
        sections = {u'train': train_sections, u'test': test_sections, u'dev': dev_sections}

    print (u'Reading train set . . .')
    print (u'train sections:', sections[u'train'])
    train_set = read_sections(pdtb_data_dir_path, sections[u'train'])
    print (u'Number of documents in train_set: ', len(train_set))
    print (u'Reading dev set . . .')
    print (u'dev sections:', sections[u'dev'])
    dev_set = read_sections(pdtb_data_dir_path, sections[u'dev'])
    print (u'Number of documents in dev_set: ', len(dev_set))
    print (u'Reading test set . . .')
    print (u'test sections:', sections[u'test'])
    test_set = read_sections(pdtb_data_dir_path, sections[u'test'])
    print (u'Number of documents in test_set: ', len(test_set))

    pickle.dump((train_set, dev_set, test_set), open(os.path.join(output_dir, "pdtb_v2_relations_implicit_explicit.p"), "wb"))


def main(args):

    verbose = False
    read_pdtb(args['pdtb_data_dir_path'], args['output_dir'], args['task_type'])


def parse_cmd_args():

    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('pdtb_data_dir_path', help = 'PDTB data directory')
    parser.add_argument('task_type', help = 'Task type is 4_way, 11_way or 15_way')
    parser.add_argument('output_dir', help = 'Output directory where parsed output should be saved')
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':

    main(parse_cmd_args())
