import codecs
import os
import logging

def bleu(reference, candidate, log_path, print_log, config):
    ref_file = log_path+'reference.txt'
    cand_file = log_path+'candidate.txt'
    with codecs.open(ref_file, 'w', 'utf-8') as f:
        for s in reference:
            if not config.char:
                f.write(" ".join(s) + '\n')
            else:
                f.write("".join(s) + '\n')
    with codecs.open(cand_file, 'w', 'utf-8') as f:
        for s in candidate:
            if not config.char:
                f.write(" ".join(s).strip() + '\n')
            else:
                f.write("".join(s).strip() + '\n')

    if config.refF != '':
        ref_file = config.refF

    temp = log_path + "result.txt"
    command = "perl script/multi-bleu.perl " + ref_file + "<" + cand_file + "> " + temp
    os.system(command)
    with open(temp) as ft:
        result = ft.read()
    os.remove(temp)
    print_log(result)

    return float(result.split()[2][:-1])

import pyrouge
def rouge(reference, candidate, log_path, print_log, config):
    assert len(reference) == len(candidate)
    reference, id2word = chinese2id(reference, {})
    candidate, _ = chinese2id(candidate, id2word)
    print(reference[:1])
    print(candidate[:1])
    ref_dir = log_path + 'reference/'
    cand_dir = log_path + 'candidate/'
    if not os.path.exists(ref_dir):
        os.mkdir(ref_dir)
    if not os.path.exists(cand_dir):
        os.mkdir(cand_dir)

    for i in range(len(reference)):
        with codecs.open(ref_dir+"%06d_reference.txt" % i, 'w', 'utf-8') as f:
            f.write("".join(reference[i]).replace(' <\s> ', '\n') + '\n')
        with codecs.open(cand_dir+"%06d_candidate.txt" % i, 'w', 'utf-8') as f:
            f.write("".join(candidate[i]).replace(' <\s> ', '\n').replace('<unk>', 'UNK') + '\n')

    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_candidate.txt'
    r.model_dir = ref_dir
    r.system_dir = cand_dir
    logging.getLogger('global').setLevel(logging.WARNING)
    rouge_results = r.convert_and_evaluate()
    scores = r.output_to_dict(rouge_results)
    recall = [round(scores["rouge_1_recall"] * 100, 2),
              round(scores["rouge_2_recall"] * 100, 2),
              round(scores["rouge_l_recall"] * 100, 2)]
    precision = [round(scores["rouge_1_precision"] * 100, 2),
                 round(scores["rouge_2_precision"] * 100, 2),
                 round(scores["rouge_l_precision"] * 100, 2)]
    f_score = [round(scores["rouge_1_f_score"] * 100, 2),
               round(scores["rouge_2_f_score"] * 100, 2),
               round(scores["rouge_l_f_score"] * 100, 2)]
    print("F_measure: %s Recall: %s Precision: %s\n"
          % (str(f_score), str(recall), str(precision)))
    return f_score[:], recall[:], precision[:]

def chinese2id(texts, word2id):
    nts = []
    for text in texts:
        nt = []
        if isinstance(text, list):
            text = ' '.join(text)
            text = text.replace('  ', ' ')
        for j in text.split(' '):
            # j = keep(j)
            if len(j) > 0:
                if j in word2id:
                    nt.append(word2id[j])
                else:
                    word2id[j] = str(len(word2id))
                    nt.append(word2id[j])
            else:
                continue
        nts.append(' '.join(nt))
    return nts, word2id

if __name__ == "__main__":
    basepath = '/home/usts/gzzh/PyCharm/TextSum/LCSTS_Log/0.1_16/'
    with open(basepath+'candidate.txt', 'r') as f:
        cand = f.readlines()
    cand = [' '.join(list(c)) for c in cand]
    with open(basepath+'reference.txt', 'r') as f:
        ref = f.readlines()
    print(len(cand), len(ref))
    ref, id2word = chinese2id(ref, {})
    cand, _ = chinese2id(cand, id2word)
    a = rouge(ref, cand, './', './', None)
    print(a)
