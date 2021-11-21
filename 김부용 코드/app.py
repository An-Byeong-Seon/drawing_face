from flask import Flask, render_template, request

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

    if k == True:
        sourceLang = 'ko'
        targetLang = 'en'

        en_text = nt.naver_translation(input_text, sourceLang, targetLang, naver_api_id, naver_api_secret) # 번역된 영어 텍스트
        e_shape_pred = 1
        f_shape_pred = 1
        h_curl_pred = 1
        h_bang_pred = 1
        h_length_pred = 1
        nose_pred = 1
        sex_pred = 1
    
    
    else:
        en_text = input_text
        e_shape_pred = 0
        f_shape_pred = 0
        h_curl_pred = 0
        h_bang_pred = 0
        h_length_pred = 0
        nose_pred = 0
        sex_pred = 0
        
    #predict(en_text)

    result = ''.join(str(_) for _ in [e_shape_pred, f_shape_pred, h_curl_pred, h_bang_pred, h_length_pred, nose_pred, sex_pred])

    face_maker(e_shape_pred, f_shape_pred, h_curl_pred,
                h_bang_pred, h_length_pred, nose_pred, sex_pred)

    return render_template('index.html', data=result)

if __name__ == '__main__':
    app.run()