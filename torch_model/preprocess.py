import pandas as pd



PREPROCESS = True

if PREPROCESS:
    # read original tsv
    df = pd.read_csv('./data/draw.tsv', sep='\t')

    # preprocess - description 
    _description = df['description']
    res = []
    for i in range(_description.shape[0]):
        try:
            arr = _description[i].split('.')
            if len(arr) > 5:
                if len(arr[-1]) > 2:
                    res.append(arr)
                else:
                    res.append(arr[:-1])
        except:
            break


    # for labeling
    _df = df.copy()
    _df['h_length'] = _df['h_length'].apply(lambda x:"hl"+str(int(x))) # description X
    _df['h_bang'] = _df['h_bang'].apply(lambda x:"hb"+str(int(x)))
    _df['h_curl'] = _df['h_curl'].apply(lambda x:"hc"+str(int(x)))
    _df['e_shape'] = _df['e_shape'].apply(lambda x:"es"+str(int(x))) # description X
    _df['f_shape'] = _df['f_shape'].apply(lambda x:"fs"+str(int(x)))
    _df['sex'] = _df['sex'].apply(lambda x:"s"+str(int(x)))
    _df['nose'] = _df['nose'].apply(lambda x:"n"+str(int(x))) # description X

    # arrange data for train
    train_df = pd.DataFrame(columns=['label', 'description'])
    for i in range(len(res)):
        idx = pd.DataFrame(res[i])[0].str.contains('bang').argmax()
        train_df = train_df.append({'label' : _df['h_bang'][i], 'description' : str(res[i][idx])}, ignore_index=True)

        idx = pd.DataFrame(res[i])[0].str.contains('curl|straight').argmax()
        train_df = train_df.append({'label' : _df['h_curl'][i], 'description' : str(res[i][idx])}, ignore_index=True)

        idx = pd.DataFrame(res[i])[0].str.contains('face').argmax()
        train_df = train_df.append({'label' : _df['f_shape'][i], 'description' : str(res[i][idx])}, ignore_index=True)

        idx = pd.DataFrame(res[i])[0].str.contains('woman|man|girl|boy').argmax()
        train_df = train_df.append({'label' : _df['sex'][i], 'description' : str(res[i][idx])}, ignore_index=True)

    # shuffle - prob not necessary
    df_shuffled = train_df.sample(frac=1).reset_index(drop=True)


    # for overfitting & small dataset
    # rat 0.9 - real / rat 1 - test data in train dataset
    rat = 0.9
    
    # - train dataset
    s, e = 0, int(df_shuffled.shape[0] * rat)
    df_train = pd.DataFrame({'label' : df_shuffled['label'][s:e],
                            'description' : df_shuffled['description'][s:e]})

    # - test dataset
    if rat == 1.0: 
        rat -= 0.1
        e = int(df_shuffled.shape[0] * (rat))
    s, e = e, e + int(df_shuffled.shape[0] * (1.0 - rat))
    df_test = pd.DataFrame({'label' : df_shuffled['label'][s:e],
                            'description' : df_shuffled['description'][s:e]})

    # save data
    df_train.to_csv('./data/draw.train_real.tsv', header=False, index=False, sep='\t')
    df_test.to_csv('./data/draw.test_real.tsv', header=False, index=False, sep='\t')



