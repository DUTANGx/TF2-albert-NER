import json
import requests
import numpy as np
import tensorflow_addons as tfa
from vocab import tokenizer
from utils import to_comma_seperated_array


def ner_boson(sentence, max_len=256):
    processed_sentence = tokenizer.tokenize_to_ids(sentence, max_len)
    req = '{"inputs": %s}' % to_comma_seperated_array(
        np.array(processed_sentence))
    response = json.loads(requests.post(
        url='http://0.0.0.0:8501/v1/models/albert_ner/versions/2:predict',
        data=req).text)['outputs']
    preds = np.array(response[0])
    transition_params = np.load(
        'models/albert_crf/saved_model/2/transition.npy')
    decoded = tfa.text.viterbi_decode(preds, transition_params)[0]
    print(decoded)
    real_len = len(sentence)
    entity = {}
    entities = []

    def check_entity_over(x, pos):
        if x:
            x['end_position'] = pos - 1
            entities.append(dict(x))
            x.clear()
        else:
            pass

    # to prevent mis labelling
    def check_type_and_add(x, type_name):
        if x and x['type'] == type_name:
            x['str'] += (sentence[i])

    for i in range(0, real_len):
        if decoded[i + 1] == 1:
            check_entity_over(entity, i)
            entity = {'type': 'person', 'start_position': i,
                      'str': sentence[i]}
        elif decoded[i + 1] == 2:
            check_type_and_add(entity, 'person')
        elif decoded[i + 1] == 3:
            check_entity_over(entity, i)
            entity = {'type': 'location', 'start_position': i,
                      'str': sentence[i]}
        elif decoded[i + 1] == 4:
            check_type_and_add(entity, 'location')
        elif decoded[i + 1] == 5:
            check_entity_over(entity, i)
            entity = {'type': 'organization', 'start_position': i,
                      'str': sentence[i]}
        elif decoded[i + 1] == 6:
            check_type_and_add(entity, 'organization')
        elif decoded[i + 1] == 7:
            check_entity_over(entity, i)
            entity = {'type': 'time', 'start_position': i,
                      'str': sentence[i]}
        elif decoded[i + 1] == 8:
            check_type_and_add(entity, 'time')
        elif decoded[i + 1] == 9:
            check_entity_over(entity, i)
            entity = {'type': 'company_name', 'start_position': i,
                      'str': sentence[i]}
        elif decoded[i + 1] == 10:
            check_type_and_add(entity, 'company_name')
        elif decoded[i + 1] == 11:
            check_entity_over(entity, i)
            entity = {'type': 'product_name', 'start_position': i,
                      'str': sentence[i]}
        elif decoded[i + 1] == 12:
            check_type_and_add(entity, 'product_name')
        else:
            check_entity_over(entity, i)

    return entities


def ner(sentence, max_len=128):
    processed_sentence = tokenizer.tokenize_to_ids(sentence, max_len)
    req = '{"inputs": %s}' % to_comma_seperated_array(
        np.array(processed_sentence))
    response = json.loads(requests.post(
        url='http://0.0.0.0:8501/v1/models/albert_ner/versions/1:predict',
        data=req).text)['outputs']
    preds = np.array(response[0])
    transition_params = np.load(
        'models/albert_crf/saved_model/1/transition.npy')
    decoded = tfa.text.viterbi_decode(preds, transition_params)[0]
    real_len = len(sentence)
    entity = {}
    entities = []

    def check_entity_over(x, pos):
        if x:
            x['end_position'] = pos - 1
            entities.append(dict(x))
            x.clear()
        else:
            pass

    def check_type_and_add(x, type_name):
        if x and x['type'] == type_name:
            x['str'] += (sentence[i])

    for i in range(0, real_len):
        if decoded[i + 1] == 1:
            check_entity_over(entity, i)
            entity = {'type': 'person', 'start_position': i,
                      'str': sentence[i]}
        elif decoded[i + 1] == 2:
            check_type_and_add(entity, 'person')
        elif decoded[i + 1] == 3:
            check_entity_over(entity, i)
            entity = {'type': 'location', 'start_position': i,
                      'str': sentence[i]}
        elif decoded[i + 1] == 4:
            check_type_and_add(entity, 'location')
        elif decoded[i + 1] == 5:
            check_entity_over(entity, i)
            entity = {'type': 'organization', 'start_position': i,
                      'str': sentence[i]}
        elif decoded[i + 1] == 6:
            check_type_and_add(entity, 'organization')
        else:
            check_entity_over(entity, i)

    return entities


if __name__ == '__main__':
    sentence = '马云，祖籍浙江省绍兴嵊州市谷来镇，后父母移居杭州，出生于浙江省杭州市，' \
               '中国企业家，中国共产党党员。曾为亚洲首富、阿里巴巴集团董事局主席。'
    # sentence2 = '2000年在深圳，马化腾作为腾讯的总经理参与了微信的开发。'
    print(ner(sentence))
    print(ner_boson(sentence))
