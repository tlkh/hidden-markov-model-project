import utils
import emission
import transition
import viterbi

import copy
from random import randint, choice
import numpy as np

entity_f_list = []
sentiment_f_list = []
k_list = []

def run_k(k):
    dataset_folder = "data/EN/"
    train_data = dataset_folder + "train"
    lines = utils.read_file_to_lines(train_data)
    
    emission_data = emission.generate_emission_table(lines)
    
    hashmap = emission_data["x_hashmap"]
    word_freq = emission_data["x_word_freq"]
    smoothed_hashmap = utils.add_unk(hashmap, word_freq, k=k)
    emission_data["x_hashmap"] = smoothed_hashmap

    x_vocab = utils.get_emission_vocab(smoothed_hashmap)
    print("Vocab size:", len(x_vocab))
    
    transition_pairs = transition.generate_transition_pairs(lines)
    
    y_pairs = transition_pairs["Y_pairs"]
    y_vocab = transition_pairs["y_vocab"]
    y_freq = transition_pairs["y_freq"]
    
    transition_data = transition.generate_transition_data(y_pairs, y_vocab)
    
    hmm = viterbi.HMM()
    hmm.fit_word_tokenizer(x_vocab)
    hmm.fit_pos_tokenizer(y_vocab)
    hmm.build_transition_weights(y_freq, transition_data)
    hmm.build_emission_weights(emission_data)
    
    
    train_data = dataset_folder + "dev.in"
    lines = utils.read_file_to_lines(train_data)

    sentences = []

    while len(lines) > 1:
        sentence_break = lines.index("")
        sentence_xy = lines[:sentence_break]
        words = [utils.preprocess_text(token,
                                       lower=LOWER,
                                       norm_tense=NORM_TENSE,
                                       replace_number=REP_NUM,
                                       replace_year=REP_YEAR,
                                       replace_symbol=REP_SYM)
                 for token in sentence_xy]
        sentence = " ".join(words).strip()
        sentences.append(sentence)
        lines = lines[sentence_break+1:]
        
    new_words = []
    for line in sentences:
        for word in line.split(" "):
            if word not in x_vocab:
                new_words.append(word)

    new_words = list(set(new_words))
    new_words.sort()
    print("New words", len(new_words))
    
    # only for the progress bar!
    try:
        from tqdm import tqdm
        USE_TQDM = True
    except Exception as e:
        print(e, "TQDM import error, disable progress bar")

    if USE_TQDM:
        sentences_it = tqdm(sentences)
    else:
        sentences_it = sentences
        
    preds = []

    for line in sentences_it:
        pred = hmm.viterbi_predict(line)
        pred = hmm.pos_tokens_to_labels(pred)
        preds.append(pred)

    assert len(sentences) == len(preds)
    
    outfile = dataset_folder + "dev.p5.out"

    with open(outfile, "w") as f:
        for sentence, pred in zip(sentences, preds):
            word_array = sentence.split(" ")
            try:
                assert len(word_array) == len(pred)
                for i, word in enumerate(word_array):
                    f.write(word + " " + pred[i] +"\n")
            except:
                print(word_array)
                print(pred)
                break
            f.write("\n")
    
    gold_data = "./data/EN/dev.out"
    pred_data = "./data/EN/dev.p5.out"

    data = utils.run_eval(gold_data, pred_data)
    
    entity_f_list.append(data["entity_f"])
    sentiment_f_list.append(data["sentiment_f"])
    k_list.append(k)

    print("\n Results")
    print("Entity F:", data["entity_f"])
    print("Sentiment F:", data["sentiment_f"])
    print("\n\n")
    
    
for i in range(0, 10):
    run_k(i)
    