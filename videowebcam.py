import os
import cv2

if not os.path.exists('../result'):
    os.makedirs('../result')


cap = cv2.VideoCapture(0)

width = int(cap.get(3))  # 가로 길이 가져오기
height = int(cap.get(4))  # 세로 길이 가져오기
fps = 30
cnt = 1

fcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('result/webcam.avi', fcc, fps, (width, height))

class_num = input('lee, kim, lim, choi, s ')
file_num = int(input('처음 시작 파일명: '))

while (True):
    k = cv2.waitKey(1) & 0xFF
    ret, frame = cap.read()
    if ret:
        out.write(frame)
        cv2.imshow('frame', frame)

        if k == ord('s'):
            print("Screenshot saved...")
            cv2.imwrite('result/{}/{}.png'.format(class_num, file_num), frame, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])
            file_num += 1
            print("Screenshot is finished.")
            print("save file name :: {}.png".format(file_num))
        elif k == ord('q'):
            break
    else:
        print("Fail to read frame!")
        break

cap.release()
out.release()
cv2.destroyAllWindows()