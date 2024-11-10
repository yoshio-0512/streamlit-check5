import streamlit as st

# --- パスワード認証機能を追加 ---
# 正しいパスワードを設定
PASSWORD = "nikomarukun"

# セッションステートを使用してログイン状態を管理
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ログインフォーム
if not st.session_state.logged_in:
    st.title("ログイン")
    password = st.text_input("パスワードを入力してください", type="password")
    if st.button("ログイン"):
        if password == PASSWORD:
            st.session_state.logged_in = True
            st.success("ログイン成功！")
        else:
            st.error("パスワードが間違っています")
    st.stop()  # 認証が成功するまで、それ以降のコードを実行しない
# --- ここまで認証機能 ---


import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageOps, ExifTags
import numpy as np

# Streamlitのページ設定
st.set_page_config(page_title="配線検出アプリ", layout="wide")

# サイドバー設定
with st.sidebar:
    st.header("設定")
    confidence_threshold = st.slider("信頼度しきい値", 0.1, 1.0, 0.5, 0.05)
    image_size = st.radio("画像サイズ", [416, 640], index=0)
    model_path = st.text_input("モデルファイル", "e_meter_segadd2.pt")
    st.write("---")
    st.info("信頼度や画像サイズ、モデルファイルの設定を変更して検出精度を調整できます。")

# タイトルと説明
st.title("配線検出アプリ")
st.markdown("### 📷 画像をアップロードまたは撮影して、配線の状態を検出します")
st.write("このアプリはYOLOモデルを使用して、配線の検出と分類を行います。")

# モデルの読み込み
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"モデルの読み込みに失敗しました: {e}")
        return None

model = load_model(model_path)

# 上下端座標を求める関数
def topbottom(img):
    img = (img > 128) * 255
    rows, cols = img.shape

    # 上端
    for i in range(rows):
        for j in range(cols - 1, -1, -1):
            if img[i, j] == 255:
                img_top = (i, j)
                break
        if 'img_top' in locals():
            break

    # 下端
    for i in range(rows - 1, -1, -1):
        for j in range(cols):
            if img[i, j] == 255:
                img_bottom = (i, j)
                break
        if 'img_bottom' in locals():
            break

    return img_top, img_bottom

# 画像前処理関数
def preprocess_image(im):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(im._getexif().items())
        if exif[orientation] == 3:
            im = im.rotate(180, expand=True)
        elif exif[orientation] == 6:
            im = im.rotate(270, expand=True)
        elif exif[orientation] == 8:
            im = im.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass

    im_new = ImageOps.pad(im, (max(im.size), max(im.size)), color=(255, 255, 255))
    im_new = im_new.resize((image_size, image_size))
    return im_new

# 欠測処理関数
def handle_missing_wires(top_list, bottom_list):
    y0, x0 = zip(*sorted(top_list, key=lambda x: x[1]))

    # x間隔の計算
    xl1 = x0[1] - x0[0]
    xl2 = x0[2] - x0[1]

    # 欠測判定と補完
    if xl1 < xl2:  # 黒線欠損
        width = xl1
        topx_dummy = x0[2] - width
    else:  # 赤線欠損
        width = xl2
        topx_dummy = x0[0] + width

    # 平均y座標で補完点を追加
    y_avr = int(sum(y0) / len(y0))
    top_list = np.append(top_list, [(y_avr, topx_dummy)], axis=0)

    # 下端の補完
    y0, x0 = zip(*sorted(bottom_list, key=lambda x: x[1]))
    y_avr = int(sum(y0) / len(y0))
    k, l = [], []
    for line in x0:
        if line < topx_dummy:
            k.append(line)
        else:
            l.append(line)
    if len(k) == 1:
        btmx_dummy = k[0] - 5
    if len(l) == 1:
        btmx_dummy = l[0] + 5
    bottom_list = np.append(bottom_list, [(y_avr, btmx_dummy)], axis=0)

    return top_list, bottom_list

# 画像アップロードまたは撮影
uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "png", "jpeg"])
camera_image = st.camera_input("カメラで写真を撮影")

if uploaded_file or camera_image:
    img_source = uploaded_file if uploaded_file else camera_image

    # 画像を読み込み
    img = Image.open(img_source)
    st.image(img, caption="アップロードされた画像", use_column_width=True)

    # 前処理
    processed_img = preprocess_image(img)

    # YOLOモデルで予測
    if model:
        results = model.predict(processed_img, imgsz=image_size, conf=confidence_threshold, classes=0)

        if results[0].masks is None:
            st.error("配線の検出に失敗しました。画像を変えて再度お試しください。")
        else:
            # マスクデータ処理
            processed_data = {'coordinates': []}
            for r in results[0].masks:
                mask_img = r.data[0].cpu().numpy() * 255
                mask_img = mask_img.astype(int)
                top, bottom = topbottom(mask_img)
                processed_data['coordinates'].append((top, bottom))

            # 座標解析
            coordinates_list = processed_data['coordinates']
            connect_list = np.array(coordinates_list)

            top_list = np.array([t[0] for t in connect_list])
            bottom_list = np.array([t[1] for t in connect_list])

            # 欠測処理（3本の場合）
            if len(top_list) == 3:
                top_list, bottom_list = handle_missing_wires(top_list, bottom_list)

            # 結線判定
            sorted_top = np.array(sorted(top_list, key=lambda x: x[1]))
            sorted_bottom = np.array([tup for _, tup in sorted(zip(top_list, bottom_list), key=lambda x: x[0][1])])
            center = sum([coords[1] for coords in sorted_bottom]) / 4
            if np.all(sorted_bottom[::2, 1] < center) and np.all(sorted_bottom[1::2, 1] > center):
                st.success("✅ 正結線の可能性が高いです")
            else:
                st.error("❌ 誤結線の可能性があります")

            # 結果画像の描画
            im_array = results[0].plot(boxes=False)
            im = Image.fromarray(im_array[..., ::-1])
            draw = ImageDraw.Draw(im)
            for i, (y, x) in enumerate(sorted_top):
                x1, y1, x2, y2 = x - 5, y - 5, x + 5, y + 5
                draw.ellipse((x1, y1, x2, y2), fill=(255, 0, 255) if i < 2 else (0, 0, 255))
            for i, (y, x) in enumerate(sorted_bottom):
                x1, y1, x2, y2 = x - 5, y - 5, x + 5, y + 5
                draw.ellipse((x1, y1, x2, y2), fill=(255, 0, 255) if i < 2 else (0, 0, 255))

            st.image(im, caption="検出結果", use_column_width=True)
