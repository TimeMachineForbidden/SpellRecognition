import mediapipe as mp
import cv2

from scipy.spatial import distance as dist
from math import atan, atan2, degrees
from variable import SpecialEffect

DEFAULT_LANDMARKS_STYLE = mp.solutions.drawing_styles.get_default_pose_landmarks_style()
DEFAULT_HAND_CONNECTIONS_STYLE = mp.solutions.drawing_styles.get_default_hand_connections_style()

VISIBILITY_THRESHOLD = .8  # 身体关键点可见的确定性阈值
STRAIGHT_LIMB_MARGIN = 20  # 肢体夹角与180度之差的度数
EXTENDED_LIMB_MARGIN = .8  # 下肢长度相对于上肢长度的比例

LEG_LIFT_MIN = -30  # 低于水平的腿抬起的最小角度

ARM_CROSSED_RATIO = 2  # 腕部到对侧肘关节的最大距离，相对于嘴的宽度

MOUTH_COVER_THRESHOLD = .03  # 手遮住嘴巴的最大距离误差，相对于1

SHOULDER_COVER_THRESHOLD = .1

SQUAT_THRESHOLD = .1  # 臀部到膝盖的最大垂直距离

JUMP_THRESHOLD = .001  # 上升和下降的阈值

LEG_ARROW_ANGLE = 18  # 从垂直站立的角度，应为90的除数

FINGER_MOUTH_RATIO = 1.5  # 手张开相对于嘴的宽度的比例
"""
FRAME_HISTORY = 8：这是定义用于比较的帧的历史长度。在这个场景中，用于比较的是姿势的历史，而这个历史包含了最近的8帧。这是一个窗口大小，用于检测一些运动模式或者状态的变化。

HALF_HISTORY = int(FRAME_HISTORY / 2)：这是历史长度的一半。它用于在不同的上下文中执行一些逻辑，例如在代码中提到的短冷却期（cool off period）。在这里，HALF_HISTORY 被用来限制对某个动作的检测持续时间。

QUARTER_HISTORY = int(FRAME_HISTORY / 4)：这是历史长度的四分之一。它可能用于类似的目的，或者在其他地方进行不同的历史数据分析。
"""
FRAME_HISTORY = 8  # pose history is compared against FRAME_HISTORY recent frames
HALF_HISTORY = int(FRAME_HISTORY / 2)
QUARTER_HISTORY = int(FRAME_HISTORY / 4)

empty_frame = {
    'palmR_y': 0,
    'palmR_dy': 0,
    'hipL_y': 0,
    'hipR_y': 0,
    'hips_dy': 0,
}
last_frames = FRAME_HISTORY * [empty_frame.copy()]

frame_midpoint = (0, 0)

"""
功能性函数
"""


def is_missing(part):
    return any(joint['visibility'] < VISIBILITY_THRESHOLD for joint in part)


# 判断手臂是否交叉，根据手腕到对侧肘关节的距离判断。
def is_arm_crossed(elbow, wrist, max_dist):
    return dist.euclidean([elbow['x'], elbow['y']], [wrist['x'], wrist['y']]) < max_dist


# 通过 a,b,c 三个点计算角度
def get_angle(a, b, c):
    ang = degrees(atan2(c['y'] - b['y'], c['x'] - b['x']) - atan2(a['y'] - b['y'], a['x'] - b['x']))
    return ang + 360 if ang < 0 else ang


# 首先计算肢体的夹角，然后检查是否在一条直线上
def is_limb_pointing(upper, mid, lower):
    if is_missing([upper, mid, lower]):
        return False
    limb_angle = get_angle(upper, mid, lower)
    is_in_line = abs(180 - limb_angle) < STRAIGHT_LIMB_MARGIN
    return is_in_line


# 计算肢体的方向角度。
# 可以指定 closest_degrees 参数，使得输出的角度为最接近该值的倍数。
def get_limb_direction(arm, closest_degrees=45):
    # should also use atan2 but I don't want to do more math
    dy = arm[2]['y'] - arm[0]['y']  # wrist -> shoulder
    dx = arm[2]['x'] - arm[0]['x']
    angle = degrees(atan(dy / dx))
    if dx < 0:
        angle += 180

    # collapse to nearest closest_degrees; 45 for semaphore
    mod_close = angle % closest_degrees
    angle -= mod_close
    if mod_close > closest_degrees / 2:
        angle += closest_degrees

    angle = int(angle)
    if angle == 270:
        angle = -90

    return angle


# 判断手指是否收回，通过比较手指到手掌的距离。
def is_finger_in(finger, palmL, palmR, min_finger_reach):
    dL_finger = dist.euclidean([finger['x'], finger['y']], [palmL['x'], palmL['y']])
    dR_finger = dist.euclidean([finger['x'], finger['y']], [palmR['x'], palmR['y']])
    d_finger = min(dL_finger, dR_finger)
    return d_finger < min_finger_reach


# 判断手是否握拳，通过判断三个手指是否伸出。
def is_hands_closed(thumb, forefinger, pinky, middle, ring, palmL, palmR, min_finger_reach):
    thumb_out = is_finger_in(thumb, palmL, palmR, min_finger_reach)
    forefinger_out = is_finger_in(forefinger, palmL, palmR, min_finger_reach)
    pinky_out = is_finger_in(pinky, palmL, palmR, min_finger_reach)
    middle_out = is_finger_in(middle, palmL, palmR, min_finger_reach)
    ring_out = is_finger_in(ring, palmL, palmR, min_finger_reach)
    return thumb_out and forefinger_out and pinky_out and ring_out


# 抬起右脚
def is_leg_lifted(leg):
    if is_missing(leg):
        return False
    dy = leg[1]['y'] - leg[0]['y']  # knee -> hip
    dx = leg[1]['x'] - leg[0]['x']
    angle = degrees(atan2(dy, dx))
    return angle > LEG_LIFT_MIN


"""
咒语判定函数
"""


# 荧光闪烁（右手一上一下）
def is_lumin(palmR):
    global last_frames

    if is_missing([palmR]):
        return False

    last_frames[-1]['palmR_y'] = palmR['y']
    # 判断纵向移动
    if palmR['y'] > last_frames[-2]['palmR_y'] + JUMP_THRESHOLD:
        last_frames[-1]['palmR_dy'] = 1  # rising
    elif palmR['y'] < last_frames[-2]['palmR_y'] - JUMP_THRESHOLD:
        last_frames[-1]['palmR_dy'] = -1  # falling
    else:
        last_frames[-1]['palmR_dy'] = 0  # not significant dy

    # consistently rising first half, lowering second half
    up = all(frame['palmR_dy'] == 1 for frame in last_frames[:HALF_HISTORY])
    down = all(frame['palmR_dy'] == -1 for frame in last_frames[HALF_HISTORY:])

    return up and down


# 呼神护卫（张开双臂）
def is_Expecto_Patronum(armL, armR):
    left_limb_pointing = is_limb_pointing(armL[0], armL[1], armL[2])
    right_limb_pointing = is_limb_pointing(armR[0], armR[1], armR[2])
    left_and_right = is_limb_pointing(armL[1], armR[1], armR[2])
    return left_limb_pointing and right_limb_pointing and left_and_right


# 钻心剜骨（双手握拳）
def is_Crucio(hands, mouth_width, palmL, palmR):
    shift_on = len(hands) == 2
    min_finger_reach = FINGER_MOUTH_RATIO * mouth_width
    for hand in hands:
        thumb, forefinger, pinky, middle, ring = hand[4], hand[8], hand[20], hand[12], hand[16]
        hands_closed = is_hands_closed(thumb, forefinger, pinky, middle, ring, palmL, palmR, min_finger_reach)
        shift_on = shift_on and hands_closed
    return shift_on and not is_Episkey(palmL, palmR, mouth_width)


# 粉身碎骨（抬起右脚）
def is_Reducto(legL, legR):
    return not is_leg_lifted(legL) and is_leg_lifted(legR)


# 愈合如初（双手捧花）
def is_Episkey(palmL, palmR, mouth_width):
    min_finger_reach = FINGER_MOUTH_RATIO * mouth_width / 2
    return abs(palmR['x'] - palmL['x']) + abs(palmR['y'] - palmL['y']) < min_finger_reach


# 无声无息（捂嘴）
def is_Quietus(mouth, palms):
    if is_missing(palms):
        return False
    dxL = (mouth[0]['x'] - palms[0]['x'])
    dyL = (mouth[0]['y'] - palms[0]['y'])
    dxR = (mouth[1]['x'] - palms[1]['x'])
    dyR = (mouth[1]['y'] - palms[1]['y'])
    return all(abs(d) < MOUTH_COVER_THRESHOLD for d in [dxL, dyL, dxR, dyR])


# 飞来咒（跳跃）
def is_Accio(hipL, hipR):
    global last_frames

    if is_missing([hipL, hipR]):
        return False

    last_frames[-1]['hipL_y'] = hipL['y']
    last_frames[-1]['hipR_y'] = hipR['y']

    if (hipL['y'] > last_frames[-2]['hipL_y'] + JUMP_THRESHOLD) and (
            hipR['y'] > last_frames[-2]['hipR_y'] + JUMP_THRESHOLD):
        last_frames[-1]['hips_dy'] = 1  # rising
    elif (hipL['y'] < last_frames[-2]['hipL_y'] - JUMP_THRESHOLD) and (
            hipR['y'] < last_frames[-2]['hipR_y'] - JUMP_THRESHOLD):
        last_frames[-1]['hips_dy'] = -1  # falling
    else:
        last_frames[-1]['hips_dy'] = 0  # not significant dy

    # consistently rising first half, lowering second half
    jump_up = all(frame['hips_dy'] == 1 for frame in last_frames[:HALF_HISTORY])
    get_down = all(frame['hips_dy'] == -1 for frame in last_frames[HALF_HISTORY:])

    return jump_up and get_down


# 快快禁锢（双臂交叉）
def is_Colloportus(elbowL, wristL, elbowR, wristR, mouth_width):
    max_dist = mouth_width * ARM_CROSSED_RATIO
    return is_arm_crossed(elbowL, wristR, max_dist) and is_arm_crossed(elbowR, wristL, max_dist)


# 眼疾咒（捂眼睛）
def is_Conjunctivitus_Curse(eyeL, eyeR, palmL, palmR):
    dxL = (eyeL['x'] - palmL['x'])
    dyL = (eyeL['y'] - palmL['y'])
    dxR = (eyeR['x'] - palmR['x'])
    dyR = (eyeR['y'] - palmR['y'])
    return all(abs(d) < SHOULDER_COVER_THRESHOLD for d in [dxL, dyL, dxR, dyR])


# 消隐无踪（蹲下）
def is_Deletrius(hipL, kneeL, hipR, kneeR):
    if is_missing([hipL, kneeL, hipR, kneeR]):
        return False
    dyL = abs(hipL['y'] - kneeL['y'])
    dyR = abs(hipR['y'] - kneeR['y'])
    return (dyL < SQUAT_THRESHOLD) and (dyR < SQUAT_THRESHOLD)


# 盔甲护身（双臂交叉抱肩）
def is_Protego(palms, shoulderL, shoulderR):
    if is_missing([shoulderL]) and is_missing([shoulderR]):
        return False
    dxL = (shoulderL['x'] - palms[1]['x'])
    dyL = (shoulderL['y'] - palms[1]['y'])
    dxR = (shoulderR['x'] - palms[0]['x'])
    dyR = (shoulderR['y'] - palms[0]['y'])
    return all(abs(d) < MOUTH_COVER_THRESHOLD for d in [dxL, dyL, dxR, dyR])


# 阿瓦达索命（双臂举起）
def is_Avada(nose, armL, armR, mouth_width):
    left_limb_pointing = is_limb_pointing(armL[0], armL[1], armL[2])
    right_limb_pointing = is_limb_pointing(armR[0], armR[1], armR[2])
    return left_limb_pointing and right_limb_pointing and armL[2]['y'] - nose['y'] > mouth_width * 3 and armR[2]['y'] - \
        nose['y'] > mouth_width * 3


"""
主函数
"""


def recognize():
    global last_frames, frame_midpoint
    FLIP = False
    DRAW_LANDMARKS = True

    cap = cv2.VideoCapture(0)

    frame_size = (int(cap.get(3)), int(cap.get(4)))
    frame_midpoint = (int(frame_size[0] / 2), int(frame_size[1] / 2))

    with mp.solutions.pose.Pose() as pose_model:
        with mp.solutions.hands.Hands(max_num_hands=2) as hands_model:
            while cap.isOpened():
                success, image = cap.read()
                if not success: break

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pose_results = pose_model.process(image)
                hand_results = hands_model.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # 绘制身体关键点及连线
                if DRAW_LANDMARKS:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image,
                        pose_results.pose_landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS,
                        DEFAULT_LANDMARKS_STYLE)

                hands = []
                hand_index = 0
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        # 绘制手部关键点及连线
                        if DRAW_LANDMARKS:
                            mp.solutions.drawing_utils.draw_landmarks(
                                image,
                                hand_landmarks,
                                mp.solutions.hands.HAND_CONNECTIONS,
                                DEFAULT_LANDMARKS_STYLE,
                                DEFAULT_HAND_CONNECTIONS_STYLE)
                        hands.append([])
                        for point in hand_landmarks.landmark:
                            hands[hand_index].append({
                                'x': 1 - point.x,
                                'y': 1 - point.y
                            })
                        hand_index += 1
                # 是否反转（自拍视角）
                if FLIP:
                    image = cv2.flip(image, 1)  # selfie view

                if pose_results.pose_landmarks:
                    # 存储最近几帧的姿势变化情况
                    last_frames = last_frames[1:] + [empty_frame.copy()]

                    body = []
                    for point in pose_results.pose_landmarks.landmark:
                        body.append({
                            'x': 1 - point.x,
                            'y': 1 - point.y,
                            'visibility': point.visibility
                        })

                    # 嘴部参数
                    mouth = (body[9], body[10])
                    mouth_width = abs(mouth[1]['x'] - mouth[0]['x'])

                    # 手掌参数
                    palms = (body[19], body[20])
                    palmL, palmR = body[17], body[18]

                    # 腿部参数
                    legL = (body[23], body[25], body[27])  # L hip, knee, ankle
                    legR = (body[24], body[26], body[28])  # R hip, knee, ankle
                    # 膝盖
                    kneeL, kneeR = body[25], body[26]
                    # 腰部
                    hipL, hipR = body[23], body[24]

                    # 手臂参数
                    shoulderL, elbowL, wristL = body[11], body[13], body[15]
                    armL = (shoulderL, elbowL, wristL)
                    shoulderR, elbowR, wristR = body[12], body[14], body[16]
                    armR = (shoulderR, elbowR, wristR)

                    # 眼部参数
                    eyeL = body[2]
                    eyeR = body[5]

                    # 鼻子参数
                    nose = body[0]

                    # 愈合如初（双手捧花）
                    if is_Episkey(palmL, palmR, mouth_width):
                        print("愈合如初")
                        SpecialEffect.val = 1
                    # 粉身碎骨（抬起右脚）
                    elif is_Reducto(legL, legR):
                        print("粉身碎骨")
                        SpecialEffect.val = 2
                    # 钻心剜骨（双手握拳）
                    elif is_Crucio(hands, mouth_width, palmL, palmR):
                        print("钻心剜骨")
                        SpecialEffect.val = 3
                    # 呼神护卫（张开双臂）
                    elif is_Expecto_Patronum(armL, armR):
                        print("呼神护卫")
                        SpecialEffect.val = 4
                    # 无声无息（捂嘴）
                    elif is_Quietus(mouth, palms):
                        print("无声无息")
                        SpecialEffect.val = 5
                    # 消隐无踪（蹲下）
                    elif is_Deletrius(hipL, kneeL, hipR, kneeR):
                        print("消隐无踪")
                        SpecialEffect.val = 6
                    # 快快禁锢（双臂交叉）
                    elif is_Colloportus(elbowL, wristL, elbowR, wristR, mouth_width):
                        print("快快禁锢")
                        SpecialEffect.val = 7
                    # 飞来咒（跳跃）
                    elif is_Accio(hipL, hipR):
                        print("飞来飞来飞来")
                        SpecialEffect.val = 8
                    # 眼疾咒（捂眼睛）
                    elif is_Conjunctivitus_Curse(eyeL, eyeR, palmL, palmR):
                        print("眼疾咒")
                        SpecialEffect.val = 9
                    # 盔甲护身（双臂交叉抱肩）
                    elif is_Protego(palms, shoulderL, shoulderR):
                        print("盔甲咒")
                        SpecialEffect.val = 10
                    elif is_Avada(nose, armL, armR, mouth_width):
                        print("阿瓦达索命")
                        SpecialEffect.val = 11
                    # 荧光闪烁（右手一上一下）
                    elif is_lumin(palmR):
                        print("荧光闪烁")
                        SpecialEffect.val = 12

                # 将图像转换为JPEG格式
                ret, buffer = cv2.imencode('.jpg', image)
                image = buffer.tobytes()

                # 生成帧图像
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()
