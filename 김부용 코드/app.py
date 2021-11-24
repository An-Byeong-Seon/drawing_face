from flask import Flask, render_template, request
from datetime import datetime
import random
from predict import predict

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
    
    # fn = "./descriptions/"+datetime.now().strftime('%Y%m%d%H%M%S')+".tsv"

    # f = open(fn, "w", encoding="utf-8")
    # f.write("-\t"+en_text)
    # f.close()

    # vocab_size = len(en_text.replace(".", " ").split(" "))
    
    # {'TEST': -1, 's1': 0, 'fs0': 1, 'hc2': 2, 'hc0': 3, 'fs2': 4, 'hb0': 5, 'hc1': 6, 'fs1': 7, 'hb1': 8, 's0': 9, 'hb2': 10, 'fs3': 11}
    '''
    e_shape_label = {0: "올라간눈매", 1:"쳐진눈매", 2:"중간눈매", 3:"안경"}
    f_shape_label = {0: "긴얼굴", 1: "둥근", 2: "각진", 3: "역삼각"}
    h_curl_label = {0: "직모", 1: "반곱슬", 2: "곱슬"}
    h_bang_label = {0: "앞머리없음", 1: "뱅", 2: "일반앞머리"}
    h_length_label = {0: "민머리", 1: "스포츠머리", 2: "단발", 3:"장발"}
    nose_label = {0: "오똑한코", 1: "납작한코"}
    sex_label = {0: "여자", 1: "남자"}
    '''
    # predict X
    e_shape_pred = random.randrange(0,4)
    h_length_pred = random.randrange(0,4) # predict - editing
    nose_pred = random.randrange(0,2)
    
    # predict
    f_shape_pred = 0
    h_curl_pred = 0
    h_bang_pred = 0
    sex_pred = 0


    s1 = en_text.split('.')
    
    for s in s1:
        # except - split
        if len(s) < 3:
            continue

        _key = predict(s)[0]
        key = _key[:-1]
        lv = int(_key[-1])

        if key == 's':
            sex_pred = lv
            print("sex_pred : ", _key)
        elif key == 'fs':
            f_shape_pred = lv
            print("f_shape_pred : ", _key)
        elif key == 'hc':
            h_curl_pred = lv
            print("h_curl_pred : ", _key)
        elif key == 'hb':
            h_bang_pred = lv
            print("h_bang_pred : ", _key)

    print("f_shape_pred {} / h_curl_pred {} / h_bang_pred {} / sex_pred {}".format(f_shape_pred, h_curl_pred, h_bang_pred, sex_pred))


    # test - input
    '''
    He has long bangs.She has round face.He is middle-aged white man.He has brown short hair, and it's curly.
    She has long bangs.She has long face.She is middle-aged white woman.She has brown short hair, and it's curly.
    '''
    result = ''.join(str(_) for _ in [e_shape_pred, f_shape_pred, h_curl_pred, h_bang_pred, h_length_pred, nose_pred, sex_pred])

    face_maker(e_shape_pred, f_shape_pred, h_curl_pred,
                h_bang_pred, h_length_pred, nose_pred, sex_pred)
    
    print(result)

    return render_template('index.html', data=result)

if __name__ == '__main__':
    app.run()
