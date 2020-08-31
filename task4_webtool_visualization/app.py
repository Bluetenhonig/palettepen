from flask import Flask, render_template, url_for, request, redirect, send_from_directory, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_dropzone import Dropzone
import sys 
import os 

sys.path.append(r'D:\thesis\webtool_visualization')
from hiercp import make_hiercp
from flatcp import make_flatcp

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
# three forward slashes for sqlalchemy db extension is relative path, four forwards slashes is an absolute path
db = SQLAlchemy(app) #initialize db

class Todo(db.Model): 
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)
    completed = db.Column(db.Integer, default=0)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self): 
        return '<Task %r>' % self.id

@app.route('/', methods=['POST', 'GET'])
def index(): 
    # delete folder content
    image_names = os.listdir(r'D:\thesis\webtool_visualization\static\img') 
    for image in image_names: 
        os.remove(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], image))
    images = os.listdir('./static/hiercp') 
    for img in images: 
         os.remove(os.path.join('./static/hiercp', img))
    images = os.listdir('./static/flatcp') 
    for img in images: 
         os.remove(os.path.join('./static/flatcp', img))
    # list to hold our uploaded image urls
    file_urls = []   
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            # save the file with to our photos folder
            filename = photos.save(
                file,
                name=file.filename    
            )
            # append image urls
            file_urls.append(photos.url(filename))          
        return "uploading..."    
    return render_template('index.html') 

@app.route('/delete/<int:id>')
def delete(id): 
    task_to_delete = Todo.query.get_or_404(id)
    try: 
        db.session.delete(task_to_delete)
        db.session.commit()
        return redirect('/')
    except: 
        return 'There was a problem deleting that task'

@app.route('/update/<int:id>', methods=['POST','GET'])
def update(id): 
    task = Todo.query.get_or_404(id)
    if request.method == 'POST': 
        task.content = request.form['content']

        try: 
            db.session.commit()
            return redirect('/')
        except: 
            return 'There was an issue updating your task'
    else: 
        return render_template('update.html', task=task)

# upload photos
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST']= 'static/img'
configure_uploads(app, photos)
patch_request_class(app) # set max file size, default=16MB

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    # list to hold our uploaded image urls
    
    if request.method == 'POST' and 'photo' in request.files:       
        uploaded_files = request.files.getlist('photo')
        for filename in request.files.getlist('photo'):
            photos.save(filename)
            filename = filename.filename
             
    return render_template('upload.html')

@app.route('/upload/<filename>')
def show(filename):
    # photo = Photo.load(id)
    # if photo is None:
    #     abort(404)
    # url = photos.url(photo.filename)
    return send_from_directory('static/img', filename)

@app.route('/gallery')
def get_gallery():
# query database where each user puts images inside 
    image_names = os.listdir(r'D:\thesis\webtool_visualization\static\img')       
    return render_template('gallery.html', image_names=image_names)

@app.route('/gallery/<hiercp>')
def show_hiercp(hiercp):
    return send_from_directory('static/hiercp', hiercp)

@app.route('/gallery/<flatcp>')
def show_flatcp(flatcp):
    return send_from_directory('static/flatcp', flatcp)


@app.route('/get_hiercp/')
def get_hiercp():
    # mit relativen pfaden arbeiten nicht absoluten 
    image_names = os.listdir(r'D:\thesis\webtool_visualization\static\img')        
    os.chdir(r'D:\thesis\webtool_visualization')
    IMAGE_PATH = r"D:\thesis\webtool_visualization\static\img" #save image-extracted CP
    EXTENSION = '.jpg'
    IMAGE_FILE = [] #['45445.jpg', '45446.jpg', '45447.jpg', '45448.jpg']
    # glob.glob() 
    for r, d, f in os.walk(IMAGE_PATH): # r=root, d=directories, f = files
        for file in f:
            if EXTENSION in file:
                IMAGE_FILE.append(file)  
    CP_PATH =  r"D:\thesis\webtool_visualization\static\hiercp" #save image-extracted CP 

    make_hiercp(IMAGE_PATH, IMAGE_FILE, CP_PATH, EXTENSION)
    hiercp_names = os.listdir(CP_PATH) 
    return render_template('gallery.html', hiercp_names=hiercp_names, image_names=image_names)
  

@app.route('/get-flatcp/')
def get_flatcp():
    print('hello')
    os.chdir(r'D:\thesis\webtool_visualization')
    IMAGE_PATH = r"D:\thesis\webtool_visualization\static\img" #save image-extracted CP
    EXTENSION = '.jpg'
    IMAGE_FILE = [] #['45445.jpg', '45446.jpg', '45447.jpg', '45448.jpg']
    for r, d, f in os.walk(IMAGE_PATH): # r=root, d=directories, f = files
        for file in f:
            if EXTENSION in file:
                IMAGE_FILE.append(file)  
    CP_PATH =  r"D:\thesis\webtool_visualization\static\flatcp" #save image-extracted CP 

    make_flatcp(IMAGE_PATH, IMAGE_FILE, CP_PATH, EXTENSION)
    flatcp_names = os.listdir(CP_PATH) 
    return render_template('gallery.html', flatcp_names=flatcp_names)
  



#Dropzone settings
dropzone = Dropzone(app)
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'upload'


if __name__ == "__main__": 
    app.run(debug=True)