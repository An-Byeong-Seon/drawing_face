from flask import Flask, render_template, request
from datetime import datetime

from predict import predict
from predict import RNN

import naver_translate as nt
from face_maker import face_maker
#from predict import predict

app = Flask(__name__)

# naver translation
naver_api_id = '' # 개발자센터에서 발급받은 Client ID 값
naver_api_secret = '' # 개발자센터에서 발급받은 Client Secret 값

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text_input', methods=['POST'])
def text_input():

    input_text = request.form['desc'] # 한국어 묘사글

    if len(input_text)<1:
        return '묘사를 입력해주십시오.' + '<a href="/">Back home</a>'

    if len(input_text) > 200: # 묘사가 너무 길면 자른다.
        input_text = input_text[:199]

    k = False

    for c in input_text:
        if ord('가') <= ord(c) <= ord('힣'):
            k = True
            break

    if k == True: # 데이터를 영어로 학습시켰으므로 한국어가 입력되면 번역해야 함
        sourceLang = 'ko'
        targetLang = 'en'

        en_text = nt.naver_translation(input_text, sourceLang, targetLang, naver_api_id, naver_api_secret) # 번역된 영어 텍스트
    
    
    else: # 한국어가 포함되지 않는 경우
        en_text = input_text
    
    fn = "./descriptions/"+datetime.now().strftime('%Y%m%d%H%M%S')+".tsv"

    f = open(fn, "w", encoding="utf-8")
    f.write("-\t"+en_text)
    f.close()

    vocab_size = len(en_text.replace(".", " ").split(" "))

    e_shape_pred = predict('./nets/rnn_weight_e_shape.pkl', fn, vocab_size)
    f_shape_pred = predict('./nets/rnn_weight_f_shape.pkl', fn, vocab_size)
    h_curl_pred = predict('./nets/rnn_weight_h_curl.pkl', fn, vocab_size)
    h_bang_pred = predict('./nets/rnn_weight_h_bang.pkl', fn, vocab_size)
    h_length_pred = predict('./nets/rnn_weight_h_length.pkl', fn, vocab_size)
    nose_pred = predict('./nets/rnn_weight_nose.pkl', fn, vocab_size)
    sex_pred = predict('./nets/rnn_weight_sex.pkl', fn, vocab_size)

    result = ''.join(str(_) for _ in [e_shape_pred, f_shape_pred, h_curl_pred, h_bang_pred, h_length_pred, nose_pred, sex_pred])

    face_maker(e_shape_pred, f_shape_pred, h_curl_pred,
                h_bang_pred, h_length_pred, nose_pred, sex_pred)
    
    print(result)

    return render_template('index.html', data=result)

if __name__ == '__main__':
    app.run()