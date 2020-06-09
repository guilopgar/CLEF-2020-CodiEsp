### Module containing auxiliary functions and classes for NLP using BERT


## Load text

import os

def load_text_files(file_names, path):
    """
    It loads the text contained in a set of files into a returned list of strings.
    Code adapted from https://stackoverflow.com/questions/33912773/python-read-txt-files-into-a-dataframe
    """
    output = []
    for f in file_names:
        with open(path + f, "r") as file:
            output.append(file.read())
            
    return output


## Keras BERT Tokenizer

# Our aim is to use the same tokenizer the Keras BERT library applies before performing WordPiece sub-tokenization.
# For that reason, the next code is adapted from: https://github.com/CyberZHG/keras-bert/blob/master/keras_bert/tokenizer.py

import unicodedata

def is_punctuation(ch):
    code = ord(ch)
    return 33 <= code <= 47 or \
        58 <= code <= 64 or \
        91 <= code <= 96 or \
        123 <= code <= 126 or \
        unicodedata.category(ch).startswith('P')

def is_cjk_character(ch):
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
            0x3400 <= code <= 0x4DBF or \
            0x20000 <= code <= 0x2A6DF or \
            0x2A700 <= code <= 0x2B73F or \
            0x2B740 <= code <= 0x2B81F or \
            0x2B820 <= code <= 0x2CEAF or \
            0xF900 <= code <= 0xFAFF or \
            0x2F800 <= code <= 0x2FA1F

def is_space(ch):
    return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
        unicodedata.category(ch) == 'Zs'

def is_control(ch):
    return unicodedata.category(ch) in ('Cc', 'Cf')

def tokenize(text, cased=True):
    if not cased:
        text = unicodedata.normalize('NFD', text)
        text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
        text = text.lower()
    spaced = ''
    for ch in text:
        if is_punctuation(ch) or is_cjk_character(ch):
            spaced += ' ' + ch + ' '
        elif is_space(ch):
            spaced += ' '
        elif ord(ch) == 0 or ord(ch) == 0xfffd or is_control(ch):
            continue
        else:
            spaced += ch
    tokens = []
    for word in spaced.strip().split():
        # tokens += self._word_piece_tokenize(word) (Original implementation)
        tokens += [word]
    return tokens


## Stop words

def remove_sw(raw_text, stop_words):
    """
    Remove all stop words from a given text (str).
    """
    return ' '.join([word for word in tokenize(raw_text) if word.lower() not in stop_words])



# NER fragments generation

def word_piece_tokenize(word, token_dict):
    """
    Code taken from: https://github.com/CyberZHG/keras-bert/blob/master/keras_bert/tokenizer.py#L121
    """
    
    if word in token_dict:
        return [word]
    tokens = []
    start, stop = 0, 0
    while start < len(word):
        stop = len(word)
        while stop > start:
            sub = word[start:stop]
            if start > 0:
                sub = '##' + sub
            if sub in token_dict:
                break
            stop -= 1
        if start == stop:
            stop += 1
        tokens.append(sub)
        start = stop
    return tokens


def start_end_tokenize(text, token_dict, cased=True):
    """
    Code adapted from: https://github.com/CyberZHG/keras-bert/blob/master/keras_bert/tokenizer.py#L101
    """
    
    if not cased:
        text = unicodedata.normalize('NFD', text)
        text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
        text = text.lower()
    spaced = ''
    start_i = 0
    start_arr = []
    for ch in text:
        if is_punctuation(ch) or is_cjk_character(ch):
            spaced += ' ' + ch + ' '
            start_arr.append(start_i)
        elif is_space(ch):
            spaced += ' '
        elif ord(ch) == 0 or ord(ch) == 0xfffd or is_control(ch):
            continue
        else:
            spaced += ch
            start_arr.append(start_i)
        start_i += 1
    
    start_end_arr, tokens = [], []
    i = 0
    for word in spaced.strip().split():
        start_i = start_arr[i]
        end_i = start_arr[i + len(word) - 1] + 1
        i += len(word)
        sub_tokens = word_piece_tokenize(word, token_dict)
        tokens += sub_tokens
        start_end_arr += [(start_i, end_i)]*len(sub_tokens)
        
    return tokens, start_end_arr


import numpy as np
import pandas as pd

def process_ner_labels(df_ann):
    df_res = []
    for i in range(df_ann.shape[0]):
        ann_i = df_ann.iloc[i].values
        # Separate discontinuous locations and split each location into start and end offset
        ann_loc_i = ann_i[4]
        for loc in ann_loc_i.split(';'):
            split_loc = loc.split(' ')
            df_res.append(np.concatenate((ann_i[:4], [int(split_loc[0]), int(split_loc[1])])))

    return pd.DataFrame(np.array(df_res), 
                        columns=list(df_ann.columns[:-1]) + ["start", "end"]).drop_duplicates()


## Custom heuristic

def convert_token_to_id_segment(token_list, tokenizer, seq_len):
    """
    Given a list of tokens representing a sentence, and a tokenizer, it returns their correponding lists of 
    indices and segments. Padding is added as appropriate.
    
    Code adapted from https://github.com/CyberZHG/keras-bert/tree/master/keras_bert/tokenizer.py#L72
    """
    
    # Add [CLS] and [SEP] tokens (second_len = 0)
    tokens, first_len, second_len = tokenizer._pack(token_list, None)
    # Generate idices and segments
    token_ids = tokenizer._convert_tokens_to_ids(tokens)
    segment_ids = [0] * first_len + [1] * second_len
    
    # Padding
    pad_len = seq_len - first_len - second_len
    token_ids += [tokenizer._pad_index] * pad_len
    segment_ids += [0] * pad_len

    return token_ids, segment_ids


from tqdm import tqdm

def heur_create_frag_input_data(df_text, text_col, df_ann, doc_list, tokenizer, lab_encoder, seq_len):
    """
    df_ann columns expected: code, start, end
    """
    indices, segments, labels, fragments, start_end_offsets = [], [], [], [], []
    for doc in tqdm(doc_list):
        # Extract doc annotation
        doc_ann = df_ann[df_ann["doc_id"] == doc].sort_values(by=["start", "end"]) # sort by start and end offset when implementing heuristic
        # Tokenize doc text into a list of sub-tokens and generate start-end offset for each sub-token
        doc_token, doc_start_end = start_end_tokenize(text=df_text[df_text["doc_id"] == doc][text_col].values[0], 
                                                      token_dict=tokenizer._token_dict)
        # Split the list of sub-tokens (sample) into fragments, and convert sub-token fragments into indices & segments
        n_frag = 0
        # Also, associate to each fragment the labels (codes) from the NER-annotations exclusively occurring inside 
        # the fragment
        ann_sup = 0
        ann_prev = 0
        for i in range(0, len(doc_token), seq_len-2):
            n_frag += 1
            # Indices & Segments
            frag_token = doc_token[i:i+seq_len-2]
            frag_id, frag_seg = convert_token_to_id_segment(frag_token, tokenizer, seq_len)
            indices.append(frag_id)
            segments.append(frag_seg)
            # Labels
            frag_start_end = doc_start_end[i:i+seq_len-2]
            start_end_offsets.append(frag_start_end)
            # Heuristic
            p = 0
            while p < ann_prev and frag_start_end[0][1] > doc_ann.iloc[ann_sup + p]['end']:
                p += 1
            ann_sup += p
            j = ann_sup
            while j < doc_ann.shape[0] and doc_ann.iloc[j]['start'] < frag_start_end[-1][1]:
                j += 1
            labels.append(list(doc_ann.iloc[ann_sup:j]['code'].values))
            if j > 0:
                ann_prev = sum(doc_ann.iloc[ann_sup:j]['start'] == doc_ann.iloc[j-1]['start'])
                ann_sup = j-ann_prev
            
        # Store the number of fragments of each doc text
        fragments.append(n_frag)
        
    # Indices & Segments shape: n_doc*n_frag x seq_len, where n_frag vary for each sample (doc)
    # Labels shape: n_doc*n_frag x n_classes
    # Fragments shape: n_doc
    return np.array(indices), np.array(segments), lab_encoder.transform(labels), np.array(fragments), start_end_offsets


# As all abstracts have one single fragment, text-classiification method is compatible with NER method to generate data

def create_frag_input_data(df_text, text_col, df_label, doc_list, tokenizer, lab_encoder, seq_len):
    indices, segments, labels, fragments = [], [], [], []
    for doc in tqdm(doc_list):
        # Extract labels
        doc_labels = list(df_label[df_label["doc_id"] == doc]["code"])
        # Tokenize doc text into a list of tokens
        doc_token = tokenizer._tokenize(df_text[df_text["doc_id"] == doc][text_col].values[0])
        # Split the list of tokens (sample) into fragments, and convert token fragments into indices & segments
        n_frag = 0
        for i in range(0, len(doc_token), seq_len-2):
            n_frag += 1
            frag_token = doc_token[i:i+seq_len-2]
            frag_id, frag_seg = convert_token_to_id_segment(frag_token, tokenizer, seq_len)
            indices.append(frag_id)
            segments.append(frag_seg)
            labels.append(doc_labels)
            # Store the number of fragments of each doc text
        fragments.append(n_frag)
        
    # Indices & Segments shape: n_doc*n_frag x seq_len, where n_frag vary for each sample (doc)
    # Labels shape: n_doc*n_frag x n_classes
    # Fragments shape: n_doc
    return np.array(indices), np.array(segments), lab_encoder.transform(labels), np.array(fragments)



# MAP score evaluation

def max_fragment_prediction(y_frag_pred, n_fragments, label_encoder_classes, doc_list):
    """
    Convert fragment-level to doc-level predictions, usin max criterion.
    """
    y_pred = []
    i_frag = 0
    for n_frag in n_fragments:
        y_pred.append(y_frag_pred[i_frag:i_frag+n_frag].max(axis=0))
        i_frag += n_frag
    return prob_codiesp_prediction_format(np.array(y_pred), label_encoder_classes, doc_list)


def prob_codiesp_prediction_format(y_pred, label_encoder_classes, doc_list):
    """
    Given a matrix of predicted probabilities (m_docs x n_codes), for each document, this procedure stores all the
    codes sorted according to their probability values in descending order. Finally, predictions are saved in a dataframe
    defined following CodiEsp submission format (see https://temu.bsc.es/codiesp/index.php/2020/02/06/submission/).
    """
    
    # Sanity check
    assert y_pred.shape[0] == len(doc_list)
    
    pred_doc, pred_code, pred_rank = [], [], []
    for i in range(y_pred.shape[0]):
        pred = y_pred[i]
        # Codes are sorted according to their probability values in descending order
        codes_sort = [label_encoder_classes[j] for j in np.argsort(pred)[::-1]]
        pred_code += codes_sort
        pred_doc += [doc_list[i]]*len(codes_sort)
        # For compatibility with format_predictions function
        pred_rank += list(range(1, len(codes_sort)+1))
            
    # Save predictions in CodiEsp submission format
    return pd.DataFrame({"doc_id": pred_doc, "code": pred_code, "rank": pred_rank})


# Code adapted from: https://github.com/TeMU-BSC/CodiEsp-Evaluation-Script/blob/master/codiespD_P_evaluation.py

from trectools import TrecQrel, TrecRun, TrecEval

    
def format_predictions(pred, output_path, valid_codes, 
                       system_name = 'xx', pred_names = ['query','docid', 'rank']):
    '''
    DESCRIPTION: Add extra columns to Predictions table to match 
    trectools library standards.
        
    INPUT: 
        pred: pd.DataFrame
                Predictions.
        output_path: str
            route to TSV where intermediate file is stored
        valid_codes: set
            set of valid codes of this subtask

    OUTPUT: 
        stores TSV files with columns  with columns ['query', "q0", 'docid', 'rank', 'score', 'system']
    
    Note: Dataframe headers chosen to match library standards.
          More informative INPUT headers would be: 
          ["clinical case","code"]

    https://github.com/joaopalotti/trectools#file-formats
    '''
    # Rename columns
    pred.columns = pred_names
    
    # Not needed to: Check if predictions are empty, as all codes sorted by prob, prob-thr etc., are returned
    
    # Add columns needed for the library to properly import the dataframe
    pred['q0'] = 'Q0'
    pred['score'] = float(10) 
    pred['system'] = system_name 
    
    # Reorder and rename columns
    pred = pred[['query', "q0", 'docid', 'rank', 'score', 'system']]
    
    # Not needed to Lowercase codes
    
    # Not needed to: Remove codes predicted twice in the same clinical case
    
    # Not needed to: Remove codes predicted but not in list of valid codes
    
    # Write dataframe to Run file
    pred.to_csv(output_path, index=False, header=None, sep = '\t')


def compute_map(valid_codes, pred, gs_out_path=None):
    """
    Custom function to compute MAP evaluation metric. 
    Code adapted from https://github.com/TeMU-BSC/CodiEsp-Evaluation-Script/blob/master/codiespD_P_evaluation.py
    """
    
    # Input args default values
    if gs_out_path is None: gs_out_path = './intermediate_gs_file.txt' 
    
    pred_out_path = './intermediate_predictions_file.txt'
    ###### 2. Format predictions as TrecRun format: ######
    format_predictions(pred, pred_out_path, valid_codes)
    
    
    ###### 3. Calculate MAP ######
    # Load GS from qrel file
    qrels = TrecQrel(gs_out_path)

    # Load pred from run file
    run = TrecRun(pred_out_path)

    # Calculate MAP
    te = TrecEval(run, qrels)
    MAP = te.get_map(trec_eval=False) # With this option False, rank order is taken from the given document order
    
    ###### 4. Return results ######
    return MAP


# Code copied from: https://github.com/TeMU-BSC/CodiEsp-Evaluation-Script/blob/master/codiespD_P_evaluation.py

def format_gs(filepath, output_path=None, gs_names = ['qid', 'docno']):
    '''
    DESCRIPTION: Load Gold Standard table.
    
    INPUT: 
        filepath: str
            route to TSV file with Gold Standard.
        output_path: str
            route to TSV where intermediate file is stored
    
    OUTPUT: 
        stores TSV files with columns ["query", "q0", "docid", "rel"].
    
    Note: Dataframe headers chosen to match library standards. 
          More informative headers for the INPUT would be: 
          ["clinical case","label","code","relevance"]
    
    # https://github.com/joaopalotti/trectools#file-formats
    '''
    # Input args default values
    if output_path is None: output_path = './intermediate_gs_file.txt' 
    
    # Check GS format:
    check = pd.read_csv(filepath, sep='\t', header = None, nrows=1)
    if check.shape[1] != 2:
        raise ImportError('The GS file does not have 2 columns. Then, it was not imported')
    
    # Import GS
    gs = pd.read_csv(filepath, sep='\t', header = None, names = gs_names)  
        
    # Preprocessing
    gs["q0"] = str(0) # column with all zeros (q0) # Columnn needed for the library to properly import the dataframe
    gs["rel"] = str(1) # column indicating the relevance of the code (in GS, all codes are relevant)
    gs.docno = gs.docno.str.lower() # Lowercase codes
    gs = gs[['qid', 'q0', 'docno', 'rel']]
    
    # Remove codes predicted twice in the same clinical case 
    # (they are present in GS because one code may have several references)
    gs = gs.drop_duplicates(subset=['qid','docno'],  
                            keep='first')  # Keep first of the predictions

    # Write dataframe to Qrel file
    gs.to_csv(output_path, index=False, header=None, sep=' ')


from keras.callbacks import Callback

class EarlyMAP_Frag(Callback):
    """
    Custom callback that performs early-stopping strategy monitoring MAP-prob metric on validation fragment dataset.
    Both train and validation MAP-prob values are reported in each epoch.
    """
    
    def __init__(self, x_train, x_val, frag_train, frag_val, label_encoder_cls, valid_codes, train_doc_list, val_doc_list, 
                 train_gs_file=None, val_gs_file=None, patience=10):
        self.X_train = x_train
        self.X_val = x_val
        self.frag_train = frag_train
        self.frag_val = frag_val
        self.label_encoder_cls = label_encoder_cls
        self.valid_codes = valid_codes
        self.train_doc_list = train_doc_list
        self.val_doc_list = val_doc_list
        self.train_gs_file = train_gs_file
        self.val_gs_file = val_gs_file
        self.patience = patience
    
    
    def on_train_begin(self, logs=None):
        self.best = 0.0
        self.wait = 0
        self.best_weights = None


    def on_epoch_end(self, epoch, logs={}):
        # Metrics reporting
        ## MAP-prob
        ### Train data
        y_pred_train = self.model.predict(self.X_train)
        # Save predictions file in CodiEsp format
        df_pred_train = max_fragment_prediction(y_frag_pred=y_pred_train, n_fragments=self.frag_train, 
                                                label_encoder_classes=self.label_encoder_cls, 
                                                doc_list=self.train_doc_list)
        map_train = compute_map(valid_codes=self.valid_codes, pred=df_pred_train, gs_out_path=self.train_gs_file)
        logs['map'] = map_train
        
        ### Val data
        y_pred_val = self.model.predict(self.X_val)
        # Save predictions file in CodiEsp format
        df_pred_val = max_fragment_prediction(y_frag_pred=y_pred_val, n_fragments=self.frag_val, 
                                              label_encoder_classes=self.label_encoder_cls, 
                                              doc_list=self.val_doc_list)
        map_val = compute_map(valid_codes=self.valid_codes, pred=df_pred_val, gs_out_path=self.val_gs_file)
        logs['val_map'] = map_val
        
        print('\rmap: %s - val_map: %s' % 
              (str(round(map_train,4)),
               str(round(map_val,4))),end=100*' '+'\n')
            
        
        # Early-stopping
        if (map_val > self.best):
            self.best = map_val
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
    
    
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)



class EarlyMAP_Frag_Abstracts(Callback):
    """
    Custom callback that performs early-stopping strategy monitoring MAP-prob metric on validation fragment dataset.
    Both train and validation MAP-prob values are reported in each epoch. Also, loss is computed only on the train 
    corpus data at the end of every epoch.
    """
    
    def __init__(self, x_train, y_train, x_val, frag_train, frag_val, label_encoder_cls, valid_codes, train_doc_list, 
                 val_doc_list, batch_size, train_gs_file=None, val_gs_file=None, patience=10):
        self.X_train = x_train
        self.y_train = y_train
        self.X_val = x_val
        self.frag_train = frag_train
        self.frag_val = frag_val
        self.label_encoder_cls = label_encoder_cls
        self.valid_codes = valid_codes
        self.train_doc_list = train_doc_list
        self.val_doc_list = val_doc_list
        self.batch_size = batch_size
        self.train_gs_file = train_gs_file
        self.val_gs_file = val_gs_file
        self.patience = patience
    
    
    def on_train_begin(self, logs=None):
        self.best = 0.0
        self.wait = 0
        self.best_weights = None


    def on_epoch_end(self, epoch, logs={}):
        # Metrics reporting
        ## Loss
        ### Train-data
        loss_train = self.model.evaluate(x=self.X_train, y=self.y_train, batch_size=self.batch_size, verbose=0)
        logs["loss_train"] = loss_train
        
        ## MAP-prob
        ### Train data
        y_pred_train = self.model.predict(self.X_train)
        # Save predictions file in CodiEsp format
        df_pred_train = max_fragment_prediction(y_frag_pred=y_pred_train, n_fragments=self.frag_train, 
                                                label_encoder_classes=self.label_encoder_cls, 
                                                doc_list=self.train_doc_list)
        map_train = compute_map(valid_codes=self.valid_codes, pred=df_pred_train, gs_out_path=self.train_gs_file)
        logs['map'] = map_train
        
        ### Val data
        y_pred_val = self.model.predict(self.X_val)
        # Save predictions file in CodiEsp format
        df_pred_val = max_fragment_prediction(y_frag_pred=y_pred_val, n_fragments=self.frag_val, 
                                              label_encoder_classes=self.label_encoder_cls, 
                                              doc_list=self.val_doc_list)
        map_val = compute_map(valid_codes=self.valid_codes, pred=df_pred_val, gs_out_path=self.val_gs_file)
        logs['val_map'] = map_val
        
        print('\rloss_train: %s | map: %s - val_map: %s' % 
              (str(round(loss_train,4)),str(round(map_train,4)),
               str(round(map_val,4))),end=100*' '+'\n')
        
        # Early-stopping
        if (map_val > self.best):
            self.best = map_val
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
    
    
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)
