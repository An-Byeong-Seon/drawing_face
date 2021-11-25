import json
import urllib.request

def naver_translation(text, sourceLang, targetLang, api_id, api_secret):
    client_id = api_id
    client_secret = api_secret
    
    encText = urllib.parse.quote(text)
    data = "source=" + sourceLang + "&target=" + targetLang + "&text=" + encText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    
    if(rescode==200):
        response_body = response.read()
        data = json.loads(response_body.decode("utf-8"))
        translatedText = data["message"]["result"]["translatedText"]
        return translatedText

    else:
        print("Error Code:" + rescode)