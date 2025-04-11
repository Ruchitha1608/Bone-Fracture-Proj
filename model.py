import numpy as np
import keras

class Efficient_net_b0_model():

    def __init__(self):
        
        print("Loading model into memory")

        self.model = keras.applications.EfficientNetB0(
        # model = keras.applications.EfficientNetB0(
            include_top=True,
            weights=None,
            input_tensor=None,
            # input_shape=None,
            input_shape=(224, 224, 3),
            # input_shape=(120, 120, 1),
            # input_shape=(224, 224, 1),
            pooling=None,
            classes=1,
            classifier_activation="sigmoid",
            # classifier_activation="relu",
            name="efficientnetb0",
        )

        self.model.load_weights('./b0_highest_accuracy_weights.weights.h5')
        # model.load_weights('./b0_highest_accuracy_weights.weights.h5')

        print("Model Loading Complete")
            
    def predictImage(self,filePath):
        # # img = imread(filePath)
        # image_path = './00004541.png'
        # img = np.array(Image.open('./00004541.png').resize((224,224)))
        image = keras.utils.load_img(filePath, target_size=(224,224))
        input_arr = keras.utils.img_to_array(image)
        # input_arra = np.array(input_arr)
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        # print(input_arr)

        prediction = self.model.predict(input_arr)
        
        if prediction[0][0]>0.5:
            prediction = 1
        else:
            prediction = 0
        
        if(prediction==0):
            return "FRACTURED"

        return "NOT FRACTURED"

# current_model = Efficient_net_b0_model()

# current_model.predictImage('./00004541.png')
