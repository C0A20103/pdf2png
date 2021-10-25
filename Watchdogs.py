#!/usr/bin/env python3

import os
import subprocess
import time
from datetime import datetime
from threading import Event, Thread

import cv2
import numpy as np
import PySimpleGUI as sg


def record():
    global cap, event, out, recFlg
    fps = 30
    didTime = time.time()
    setTime = time.time()
    while True:
        event.wait()
        if (True in recFlg[:2]):
            setTime = time.time() + 3
            recFlg[2] = True
        else:
            recFlg[2] = False
        while (setTime > time.time() and event.is_set()):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame,
                        datetime.now().strftime('%Y/%m/%d %H:%M:%S.%f')[:22],
                        (0, 25),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA)
            out.write(frame)

            # フレームレート制限
            sleepTime = 1/fps - (time.time() - didTime)
            if (sleepTime > 0):
                time.sleep(sleepTime)
            didTime = time.time()

        # フレームレート制限
        sleepTime = 1 / fps - (time.time() - didTime)
        if (sleepTime > 0):
            time.sleep(sleepTime)
        didTime = time.time()


def main():
    global cap, event, out, recFlg

    # カメラ初期設定
    cd = __file__.replace(os.path.basename(__file__), '')
    cascadePath = os.path.join(cd, 'haarcascade_frontalface_alt2.xml')
    cascade = cv2.CascadeClassifier(cascadePath)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg = gray.copy().astype('float')
    frame = np.zeros((frame.shape[0], frame.shape[1], 3))
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()

    # GUI初期設定
    sg.theme('Default1')
    layout = [
            [sg.Column([[sg.Image(data=imgbytes, key='image')]], justification='center')],
            [sg.Column([[sg.Button('録画', size=(10, 1)), sg.Button('開く', size=(10, 1))]],justification='center')]
        ]
    window = sg.Window('Watchdog', layout, resizable=True)

    # 録画初期設定
    event = Event()
    thread = Thread(target=record)
    thread.start()

    # 変数初期設定
    mainFps = 10
    didTime = 0
    recFlg = [False, False, False]
    sleepTime = 0
    saveFolder = os.getcwd() + os.sep + 'capture' + os.sep
    if (not os.path.exists(saveFolder)):
        os.mkdir(saveFolder)
    while True:
        btnEvent, values = window.read(timeout=0)
        if (btnEvent != '__TIMEOUT__'):
            if (btnEvent in (None, 'Exit')):
                break
            elif (btnEvent == '録画'):
                if (window['録画'].get_text() == '録画'):
                    window['録画'].update('停止')

                    # 録画開始設定
                    out = cv2.VideoWriter(saveFolder + datetime.now().strftime('%y%m%d_%H%M%S.mp4'),
                                          cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                          30,
                                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                    event.set()
                else:
                    window['録画'].update('録画')

                    # 録画終了設定
                    event.clear()
                    out.release()
                    recFlg[2] = False
            elif(btnEvent == '開く'):
                for i in ['xdg-open', 'open', 'explorer']:
                    try:
                        subprocess.call([i, saveFolder])
                        break
                    except:
                        pass

        # カメラから画像取得
        ret, frame = cap.read()

        # 動体検知
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.accumulateWeighted(gray, avg, 0.6)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in contours:
            area = cv2.contourArea(i)
            if (area > 1500):
                recFlg[0] = True
                break
        else:
            recFlg[0] = False
        frame = cv2.drawContours(frame, contours, -1, (255, 0, 0), 3)

        # 顔認識
        faceList = cascade.detectMultiScale(gray, minSize=(100, 100))
        recFlg[1] = True if 0 > len(faceList) else False
        for (x, y, w, h) in faceList:
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 225, 0), thickness=3)

        # 画面反転
        frame = cv2.flip(frame, 1)

        # 文字列表示
        delay = time.time() - didTime
        fps = mainFps if sleepTime > 0 else 1 / ((1 / mainFps) - sleepTime)
        displayStr = [f'delay:{max(0, delay):.3f}sec', f'rate :{fps:.2f}fps', f'face :{len(faceList)}', f'MDflg:{recFlg[0]}']
        for (i, j) in enumerate(displayStr, 1):
            cv2.putText(frame, j, (0, 25*i), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
        if recFlg[2]:
            cv2.putText(frame, '*REC', (0, 125), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)

        # ウィンドウの画像更新
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window['image'].update(data=imgbytes)

        # フレームレート制限
        sleepTime = 1 / mainFps - (time.time() - didTime)
        if (sleepTime > 0):
            time.sleep(sleepTime)
        didTime = time.time()
    thread.join(timeout=0)
    cap.release()
    window.close()
    exit()


if __name__ == '__main__':
    main()