import cv2
import numpy as np

# This function can resize to any shape, but was built to resize to 84x84
def process_frame(frame, shape=(84, 84)):
    """Preprocesses a 210x160x3 frame to 84x84x1 grayscale
    Arguments:
        frame: The frame to process.  Must have values ranging from 0-255
    Returns:
        The processed frame
    """

    # cv2를 통한 데이터 감소를 이용하기 위해선 반드시 데이터 형을 np.uint8로 만들어야함
    frame = frame.astype(np.uint8) 
    
    # RGB -> GRAY로 색 변경
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # 프레임을 crop한다 ([0~34, 160 ~ 160+34] 범위 삭제 )
    frame = frame[34:34+160, :160]

    # 크기를 인자로 전달된 shape로 리사이징 한다
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)

    # reshape한다
    frame = frame.reshape((*shape, 1))

    return frame