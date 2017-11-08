import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug.utils import secure_filename
from wand.image import Image as wandimage
from skimage import exposure
from app import app
from PIL import Image, ImageFont, ImageDraw
from wand.image import Image as wandimage
from wand.api import library
from wand.image import COMPRESSION_TYPES as compression_type

global loc
global crop
global upload_folder
global basedir

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = 'uploads/'
upload_folder = os.path.join(basedir, app.config['UPLOAD_FOLDER'])

#app.static_folder = '/app/uploads'

app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'tiff'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['POST'])
def upload():
    # Get the name of the uploaded file
    global upload_folder
    file = request.files['file']
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        #print filename
        if filename.rsplit('.',1)[1] == 'pdf':
            new_filename = 'nach' + '.' + filename.rsplit('.',1)[1]
        else:
            new_filename = 'converted_nach_page' + '.' + filename.rsplit('.',1)[1]
        print new_filename
        # Move the file form the temporal folder to
        # the upload folder we setup
        file.save(upload_folder+ new_filename)
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file
        return redirect(url_for('uploaded_file', filename=new_filename))

    # Check if the file is one of the allowed types/extensions

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    global upload_folder
    print filename + "bunny"
    print os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_from_directory(upload_folder,
                               filename)

@app.route("/")
def main():
    return render_template('index.html')


@app.route('/convert')
def convert():
    with wandimage(filename='/home/sodel/Desktop/FlaskApp/app/uploads/nach.pdf', resolution=1200) as img:
        print('width =', img.width)
        print('height =', img.height)
        print('pages = ', len(img.sequence))
        print('resolution = ', img.resolution)
        with img.convert('jpeg') as converted:
           converted.save(filename=upload_folder+'converted_nach_page.jpg')
        print app.root_path
        return render_template('converted_image.html')


loc = 0
crop = 0
threshold = 0.75
@app.route('/', methods=['POST'])
def my_form_post():
    global threshold
    if request.method == 'POST':
        if request.form['my-form'] == 'Send':
            text = request.form['text']
            threshold = text
    return threshold

@app.route('/flatten_image')
def flatten_image():
    global loc
    global crop
    global threshold
    global upload_folder
    global basedir
    if os.path.exists(upload_folder + 'converted_nach_page.jpg'):
        image = cv2.imread(upload_folder + "converted_nach_page.jpg")
    else:
        image = cv2.imread(upload_folder + "converted_nach_page.jpeg")
    ratio = image.shape[0] / 800.0
    (h, w) = image.shape[:2]
    r = 800 / float(h)
    width = int(w * r)
    dim = None

    res = cv2.resize(image, (width, 800), interpolation=cv2.INTER_AREA)

    print(dim)
    print ratio

    # convert into grayscale, blur it and find its edges
    orig = image.copy()
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # use Canny edge detection method for detecting edges
    edged = cv2.Canny(blur, 75, 200)

    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        print len(approx)
        if len(approx) == 4:
            screenCnt = approx
            break

    print("finding contours of paper")
    print basedir
    pts = screenCnt.reshape(4, 2) * ratio
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    print ((tl, tr, br, bl))

    # new dimensions of image

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    X = cv2.getPerspectiveTransform(rect, dst)

    crop = cv2.warpPerspective(orig, X, (maxWidth, maxHeight))
    cv2.imwrite(upload_folder + "31.jpg", crop)
    return send_from_directory(upload_folder,
                               '31.jpg')

@app.route('/detect_edges')
def detect_edges():
    global loc
    global crop
    global threshold
    global upload_folder
    global basedir
    if os.path.exists(upload_folder + 'converted_nach_page.jpg'):
        image = cv2.imread(upload_folder + "converted_nach_page.jpg")
    else:
        image = cv2.imread(upload_folder + "converted_nach_page.jpeg")
    ratio = image.shape[0] / 800.0
    (h, w) = image.shape[:2]
    r = 800 / float(h)
    width = int(w * r)
    dim = None

    res = cv2.resize(image, (width, 800), interpolation=cv2.INTER_AREA)

    print(dim)
    print ratio

    # convert into grayscale, blur it and find its edges
    orig = image.copy()
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # use Canny edge detection method for detecting edges
    edged = cv2.Canny(blur, 75, 200)

    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        print len(approx)
        if len(approx) == 4:
            screenCnt = approx
            break

    print("finding contours of paper")
    print basedir
    pts = screenCnt.reshape(4, 2) * ratio
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    print ((tl, tr, br, bl))

    # new dimensions of image

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    X = cv2.getPerspectiveTransform(rect, dst)

    crop = cv2.warpPerspective(orig, X, (maxWidth, maxHeight))
    cv2.imwrite(upload_folder + "31.jpg", crop)

    print threshold
    #crop = image
    img_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(basedir + '/images/template.png', 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= float(threshold))
    print type(loc)
    print type(crop)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(crop, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    cv2.imwrite(upload_folder + 'res.png', crop)

    print loc
    print cv2.__version__
    pts = np.array(zip(*loc[::-1]), 'int32')
    x, y, w, h = cv2.boundingRect(pts)
    cv2.rectangle(crop, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(upload_folder + "img5_rect.jpg", crop)
    nach = crop[y:y + h, x:x + w]
    cv2.imwrite(upload_folder + 'output44t.jpg', nach)

    im = nach
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

    hull = cv2.convexHull(cnt)
    simplified_cnt = cv2.approxPolyDP(hull, 0.001 * cv2.arcLength(hull, True), True)

    width = max([i[0][0] for i in simplified_cnt])
    height = max([i[0][1] for i in simplified_cnt])
    (H, mask) = cv2.findHomography(simplified_cnt.astype('single'), np.array(simplified_cnt.astype('single')))

    final_image = cv2.warpPerspective(im, H, (width, height))
    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    final_image = exposure.rescale_intensity(final_image, out_range=(0, 255))
    cv2.imwrite(upload_folder+'output44r.jpg', final_image)
    return render_template('detect_edges.html')

def overwrite(number, x, y, font, image, output_image):
    img = Image.open('./app/uploads/' + image)
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    image_font = ImageFont.truetype("arial.ttf", font)
    draw.text((x, y), number, (0, 0, 0), font=image_font)
    img.save('./app/uploads/' + output_image)

account_number = 00000000000
account_x = 0
account_y = 0
account_font = 180
account_counter = 0
@app.route('/overwrite_account', methods=['POST'])
def overwrite_account():
    global account_number, account_x, account_y, account_font, account_counter
    account_counter = account_counter + 1
    print account_counter
    if request.method == 'POST':
        if request.form['account'] == 'Set Account':
            account_number = request.form['account_number']
            account_x = request.form['account_number_x']
            account_y = request.form['account_number_y']
            account_font = request.form['account_number_font']
            test = [account_number, account_x, account_y, account_font]
            print test
            overwrite(account_number, int(account_x), int(account_y), int(account_font), "output44r.jpg", "sample-out.jpg")
    return render_template('account_number.html')

ifsc_code = 00000000000
ifsc_code_x = 0
ifsc_code_y = 0
ifsc_code_font = 180
ifsc_counter = 0
@app.route('/overwrite_ifsc', methods=['POST'])
def overwrite_ifsc():
    global ifsc_code, ifsc_code_x, ifsc_code_y, ifsc_code_font, ifsc_counter
    ifsc_counter = ifsc_counter + 1

    if request.method == 'POST':
        if request.form['ifsc'] == 'Set IFSC':
            ifsc_code = request.form['ifsc_code']
            ifsc_code_x = request.form['ifsc_code_x']
            ifsc_code_y = request.form['ifsc_code_y']
            ifsc_code_font = request.form['ifsc_code_font']
            test = [ifsc_code, ifsc_code_x, ifsc_code_y, ifsc_code_font]
            print test
            print ifsc_counter
            overwrite(ifsc_code, int(ifsc_code_x), int(ifsc_code_y), int(ifsc_code_font), "sample-out.jpg", "sample-out1.jpg")
    return render_template('ifsc_code.html')

cdf_code = 00000000000
cdf_code_x = 0
cdf_code_y = 0
cdf_code_font = 180
cdf_counter = 0
@app.route('/overwrite_cdf', methods=['POST'])
def overwrite_cdf():
    global cdf_code, cdf_code_x, cdf_code_y, cdf_code_font, cdf_counter
    cdf_counter = cdf_counter + 1
    if request.method == 'POST':
        if request.form['cdf'] == 'Set CDF':
            cdf_code = request.form['cdf_code']
            cdf_code_x = request.form['cdf_code_x']
            cdf_code_y = request.form['cdf_code_y']
            cdf_code_font = request.form['cdf_code_font']
            test = [cdf_code, cdf_code_x, cdf_code_y, cdf_code_font]
            print test
            print cdf_counter
            overwrite(cdf_code, int(cdf_code_x), int(cdf_code_y), int(cdf_code_font), "sample-out1.jpg", "sample-out2.jpg")
    return render_template('cdf_code.html')

bank_name = 00000000000
bank_name_x = 0
bank_name_y = 0
bank_name_font = 180
bank_counter = 0
@app.route('/overwrite_bank_name', methods=['POST'])
def overwrite_bank_name():
    global bank_name, bank_name_x, bank_name_y, bank_name_font, bank_counter
    bank_counter = bank_counter + 1

    if request.method == 'POST':
        if request.form['bank'] == 'Set BANK':
            bank_name = request.form['bank_name']
            bank_name_x = request.form['bank_name_x']
            bank_name_y = request.form['bank_name_y']
            bank_name_font = request.form['bank_name_font']
            test = [bank_name, bank_name_x, bank_name_y, bank_name_font]
            print test
            print bank_counter
            overwrite(bank_name, int(bank_name_x), int(bank_name_y), int(bank_name_font), "sample-out2.jpg", "sample-out3.jpg")
    return render_template('bank_name.html')

@app.route('/final_image')
def final_image():
    if bank_counter > 0:
        print "jjjjjjjjjjjjjjjjjjjjjj"
        im = Image.open("./app/uploads/sample-out3.jpg")
    elif cdf_counter > 0:
        print "888888888888888888"
        im = Image.open("./app/uploads/sample-out2.jpg")
    elif ifsc_counter > 0:
        print "0000000000000000"
        im = Image.open("./app/uploads/sample-out1.jpg")
    elif account_counter > 0:
        print "---=-=-=-="
        im = Image.open("./app/uploads/sample-out.jpg")

    im.save(upload_folder +"output44r.jpg")
    return render_template('final_image.html')

@app.route('/compression')
def compression():
    global upload_folder
    if os.path.exists(upload_folder + "output44r.jpg"):
        im = Image.open(upload_folder + "output44r.jpg")
    elif os.path.exists(upload_folder + "31.jpg"):
        im = Image.open(upload_folder + "31.jpg")
    im = im.resize((830, 392), Image.ANTIALIAS)
    im.save(upload_folder +"kenya_buzz.jpg")
    im.save(upload_folder +"compressed_jpeg.jpg", format="JPEG", quality=70)
    nach = cv2.imread(upload_folder +'kenya_buzz.jpg')

    nach_grey = cv2.cvtColor(nach, cv2.COLOR_BGR2GRAY)
    ret, threshnach = cv2.threshold(nach_grey, 160, 255, cv2.THRESH_BINARY)
    ret2nach, th2nach = cv2.threshold(nach_grey, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th3 = cv2.adaptiveThreshold(nach_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 20)
    th2 = cv2.adaptiveThreshold(nach_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)

    ret, th10 = cv2.threshold(nach_grey, 200, 255, cv2.THRESH_BINARY)
    cv2.imwrite(upload_folder +'nach_binry.tiff', threshnach)
    cv2.imwrite(upload_folder +'nach_binry_ostu.tiff', th2nach)
    cv2.imwrite(upload_folder +'nach_binry_10.tiff', th10)
    cv2.imwrite('nach_adaptive.tiff', th3)
    cv2.imwrite('nach_guass.tiff', th2)

    with wandimage(filename='nach_guass.tiff') as img:
        library.MagickSetImageCompression(img.wand, compression_type.index('fax'))
        library.MagickSetImageCompressionQuality(img.wand, 30)
        img.save(filename=upload_folder + 'outputimage_thumbnail2kl.tiff')
    return render_template('compression.html')

if __name__ == "__main__":
    app.run()