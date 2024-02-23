import numpy as np
import pyrebase
from flask import Flask, render_template, request ,jsonify , json
import tensorflow as tf
import matplotlib as mpl
import tensorflow as tf



##--------------------------------Connection with FireBase-------------------------------------###

IMG_SIZE = 224
BATCH_SIZE = 32

def send_to_Firebase(img , imgname , storage):
     storage.child(f'/images/{imgname}').put(img)
def download_from_Firebas(path_Firebase , path_Here , storage):
     storage.download(path_Firebase, path_Here)
##--------------------------------Connection with FireBase-------------------------------------###





###-------------------------------AI functionality---------------------------------------------###
#->Loading the model Done By AI Team to use it 


def load_model(model_path):
  """
load a saved model from a path
  """
  print("Loading Saved Model")
  model = tf.keras.models.load_model(model_path)
  return model


# ->Given the image path you will Predict the label

def get_img_array(img_path, size = (224 , 224)):
    # `img` is a PIL image of size 299x299
    img = tf.keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tf.keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
def decode_predictions(preds):
  classes = ['Glioma' , 'meningioma' , 'No Tumor' , 'Pituitary']
  prediction = classes[np.argmax(preds)]
  return prediction

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4 , view = False):
    # Load the original image
    img = tf.keras.utils.load_img(img_path)
    img = tf.keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    
def make_prediction (img_path , model, last_conv_layer_name = "Top_Conv_Layer" ,
                      campath = "static\cam.jpg" ,
                        view = False):
  img = get_img_array(img_path = img_path)
  img_array = get_img_array(img_path, size=(224 , 224))
  preds = model.predict(img_array)
  heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
  save_and_display_gradcam(img_path, heatmap , cam_path=campath , view = view)
  return [campath , decode_predictions(preds)]
###-------------------------------AI functionality-------------------------------###






###------------------------------- Core API app -------------------------------###
app = Flask(__name__)
#Loading The DeepLearning Model 
#path for the testing photos
test_path = "Tumor_Detector/static"
#The Home Route
@app.route("/", methods=['GET', 'POST'])
def main():
  return render_template("index.html")
#The submit Root (The route used to predict)
@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
  firebaseConfig = {
  'apiKey': "AIzaSyCFYjB2SIgwiabbVUPFfi8ZSq4ovTDWUHY",
  'authDomain': "test-97ca6.firebaseapp.com",
  'projectId': "test-97ca6",
  'storageBucket': "test-97ca6.appspot.com",
  'messagingSenderId': "1007975514765",
  'appId': "1:1007975514765:web:fb154ed9aa3e701ad4d8c4",
  'measurementId': "G-WJ0L8LBVR0",
  "serviceAccount" : "Tumor_Detector/serviceAccount.json" , 
  "databaseURL" : "https://test-97ca6-default-rtdb.firebaseio.com/"
}
  firebase = pyrebase.initialize_app(config=firebaseConfig)
  storage = firebase.storage()
    
  if request.method == 'POST':
   img = request.files['my_image']
   img_path = "Tumor_Detector/static/" + img.filename	
   img.save("Tumor_Detector/static/" + img.filename	) 
   prediction_1 = f'{img.filename}-Prediction'
   send_to_Firebase(img_path , img.filename , storage)
   return {"prediction" : prediction_1 , 
           "FireBasePath" : f'/images/{img.filename}',
           }
   
@app.route("/submit_path", methods = ['POST'])
def get_output_path():
  firebaseConfig = {
  'apiKey': "AIzaSyCFYjB2SIgwiabbVUPFfi8ZSq4ovTDWUHY",
  'authDomain': "test-97ca6.firebaseapp.com",
  'databaseURL': "https://test-97ca6-default-rtdb.firebaseio.com",
  'projectId': "test-97ca6",
  'storageBucket': "test-97ca6.appspot.com",
  'messagingSenderId': "1007975514765",
  'appId': "1:1007975514765:web:fb154ed9aa3e701ad4d8c4",
  'measurementId': "G-WJ0L8LBVR0",
  "serviceAccount" : "Tumor_Detector/serviceAccount.json" , 
  "databaseURL" : "https://test-97ca6-default-rtdb.firebaseio.com/"
}
  firebase = pyrebase.initialize_app(config=firebaseConfig)
  storage = firebase.storage()
    
  if request.method == 'POST':
   firebase_path = str(request.form['path'])
   path_here = f"Tumor_Detector/static/{firebase_path}"
   storage.download(firebase_path, path_here)
   
   print("Downloaded")
   prediction_1 = f'{path_here}-Prediction'
   send_to_Firebase(path_here , 'Segmentation.jpeg' , storage=storage)
   data = {"prediction" : prediction_1 , 
           "FireBasePath" : f'/images/Segmentation.jpeg'}
   response = app.response_class(
     response = json.dumps(data) , 
     status=200 , 
     mimetype='application/json'
   )
   return response
  
           
 
if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
###------------------------------- Core API app -------------------------------###
