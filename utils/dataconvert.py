# coding:utf-8
import pandas as pd

with open('/home/gzzh/PyCharm/NatureLP/TextSum_LCSTS/LCSTS_ORIGIN/DATA/PART_III.txt', 'r') as f:
    a = f.readlines()
    # print(len(a)/8)
    summary = []
    short_text = []
    labels = []
    for i in range(int(len(a)/9)):
        sl_i = 9*i+1
        s_i = 9*i+3
        st_i = 9*i+6
        summary.append(a[s_i].replace(" ", ""))
        short_text.append(a[st_i].replace(" ", ""))
        labels.append(a[sl_i].replace('<human_label>', '').lstrip()[0])

data = pd.DataFrame({'summary': summary, 'short_text': short_text, 'labels': labels})
data['labels'].to_csv('datalabel.csv', sep=',', index=None)
# data['labels'] = data['labels'].astype('int32')
# data = data[data['labels'] > 2]
# print(data['labels'].value_counts())
# with open('../tmp/valid.tgt', 'w') as f:
#     for i in data['summary']:
#         f.write(' '.join(list(i)))
#     f.close()
# #
# with open('../tmp/valid.src', 'w') as f:
#     for i in data['short_text']:
#         f.write(' '.join(list(i)))
#     f.close()