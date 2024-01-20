from flask import Flask, render_template, Response
from media_pipe import recognize
from variable import SpecialEffect

app = Flask(__name__)


@app.route('/')
def index():
    # 返回HTML模板
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    # 返回图像流
    return Response(recognize(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/special_effect')
def get_special_effect():
    return Response(str(SpecialEffect.val))  # 这里需要强转类型


if __name__ == '__main__':
    app.run(debug=True)
