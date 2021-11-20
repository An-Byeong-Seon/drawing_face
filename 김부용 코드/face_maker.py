from PIL import Image
import os

def face_maker(e_shape_pred = 1, f_shape_pred = 1, h_curl_pred = 1,
                h_bang_pred = 1, h_length_pred = 1, nose_pred = 1, sex_pred = 1):

    e_shape_label = {0: "올라간눈매", 1:"쳐진눈매", 2:"중간눈매", 3:"안경"}
    f_shape_label = {0: "긴얼굴", 1: "둥근", 2: "각진", 3: "역삼각"}
    h_curl_label = {0: "직모", 1: "반곱슬", 2: "곱슬"}
    h_bang_label = {0: "앞머리없음", 1: "뱅", 2: "일반앞머리"}
    h_length_label = {0: "민머리", 1: "스포츠머리", 2: "단발", 3:"장발"}
    nose_label = {0: "오똑한코", 1: "납작한코"}
    sex_label = {0: "여자", 1: "남자"}

    e_shape_pred = e_shape_label[e_shape_pred]
    f_shape_pred = f_shape_label[f_shape_pred]
    h_curl_pred = h_curl_label[h_curl_pred]
    h_bang_pred = h_bang_label[h_bang_pred]
    h_length_pred = h_length_label[h_length_pred]
    nose_pred = nose_label[nose_pred]
    sex_pred = sex_label[sex_pred]

    root = "./static/images/"

    if os.path.isfile(root+"default2.png"):
        os.remove(root+"default2.png")

    bg_path = root + "character_bg.png"

    back_hair_path = root + "hair/뒷머리/{}형뒷머리/{}형{}{}".format(f_shape_pred, f_shape_pred, h_curl_pred, h_length_pred)
    back_hair_mask_path = back_hair_path + "마스크"

    neck_path = root + "face/목"
    neck_mask_path = neck_path + "마스크"
    neck_path2 = root + "face/남자목젖"

    f_shape_path = root + "face_shape/{}형".format(f_shape_pred)
    f_shape_mask_path = f_shape_path + "마스크"
    f_shape_path2 = root + "face/눈썹입"

    nose_path = root + "face/{}".format(nose_pred)

    e_shape_path = root + "face/{}".format(e_shape_pred)
    e_shape_path2 = root + "face/눈동자"
    e_shape_path3 = root + "face/여자속눈썹"

    h_bang_path = root + "hair/앞머리/{}".format(h_bang_pred)
    h_bang_mask_path = root + "hair/앞머리/{}형{}마스크".format(f_shape_pred, h_bang_pred)

    face_components = []

    bg = Image.open(bg_path)

    back_hair_mask = Image.open(back_hair_mask_path + ".png")
    back_hair = Image.open(back_hair_path + ".png")

    neck_mask = Image.open(neck_mask_path + ".png")
    neck = Image.open(neck_path + ".png")
    neck2 = Image.open(neck_path2 + ".png")

    f_shape = Image.open(f_shape_path + ".png")
    f_shape2 = Image.open(f_shape_path2 + ".png")
    f_shape_mask = Image.open(f_shape_mask_path + ".png")

    nose = Image.open(nose_path + ".png")

    e_shape = Image.open(e_shape_path + ".png")
    e_shape2 = Image.open(e_shape_path2 + ".png")
    e_shape3 = Image.open(e_shape_path3 + ".png")

    h_bang_mask = Image.open(h_bang_mask_path + ".png")
    h_bang = Image.open(h_bang_path + ".png")


    face_components = [bg, back_hair_mask, back_hair, neck_mask, neck, f_shape_mask, f_shape, f_shape2, nose, e_shape, e_shape2, h_bang_mask, h_bang]
    
    if sex_pred == "남자":
        face_components.append(neck2)
    else:
        face_components.append(e_shape3)

    canvas = Image.new('RGB', (500, 536), "white")

    for c in face_components:
        canvas.paste(c, (0, 0), c)

    canvas = canvas.resize((300, 322), Image.LANCZOS)
    canvas.save(root + "default2.png")