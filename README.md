# ML_ConvolucionalNN_Cat-Dog_Classification
My solution to FreeCodeCamp ["Cat-Dog-Classification"](https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/cat-and-dog-image-classifier) task: "For this challenge, you will complete the code to classify images of dogs and cats. You will use TensorFlow 2.0 and Keras to create a convolutional neural network that correctly classifies images of cats and dogs at least 63% of the time. (Extra credit if you get it to 70% accuracy!)".

## Dealing with data
After collecting the files and defining variables to create the cat and dog classification machine learning model, we have to generate the training, validation and test data.

~~~python
train_data_gen = train_image_generator.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary',
    batch_size=batch_size
)
val_data_gen = validation_image_generator.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary',
    batch_size=batch_size
)
test_data_gen = test_image_generator.flow_from_directory(
    PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    classes=['test'],
    batch_size=batch_size,
    shuffle=False
)
~~~

Since there are a small number of training examples, there is a risk of overfitting. One way to fix this problem is by creating more training data from existing training examples by using random transformations such as rotations, shifts and flips.

~~~python
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode = 'nearest'
)
~~~

## The model
The model created for classification was a Convolutional Neural Network (CNN), typical for image classification problems. It was used the Keras Sequential model. It involves a stack of Conv2D and MaxPooling2D layers, then a fully connected layer on top that is activated by a ReLU activation function and a output layer with the Sigmoid activation function to normalize the output within a 0 to 1 range.

The Adam optimizer was used, ideal for the large amount of data involved and the Binary Crossentropy loss function, as there are only two categories.

~~~python
model = Sequential()

model.add(tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(tf.keras.layers.MaxPooling2D(2))
model.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2))
model.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu', use_bias=True))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


model.compile(optimizer = 'adam',
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = ['accuracy'])
~~~

The training works as expected:

![download](https://user-images.githubusercontent.com/93870597/228310487-c66b088e-e498-42ed-bbe1-3d63268712aa.png)

## The results
The model correctly identified 72.0% of the images of cats and dogs. We can see some exemples:

![download](https://user-images.githubusercontent.com/93870597/228310952-cf9091a7-f5e8-4cba-b202-a768345ca669.png)
