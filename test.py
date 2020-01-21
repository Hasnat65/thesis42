import matplotlib.pyplot as plt
import numpy as np
import  tensorflow as tf
import keras
import train

image_width = 224
image_height = 224
image_size = (image_width, image_height)
batch_size = 10
print(tf.__version__)
image_width = 224
image_height = 224
image_size = (image_width, image_height)
batch_size = 10
df = train.generate_data()
print(df)
model = train.build_model()
model.summary()


index = np.random.randint(df.shape[0])
path = "data/" + df['file'].iloc[index]

data = plt.imread(path)
plt.imshow(data)
plt.show()

print("This is {}".format(df['file'].iloc[index].split('.')[0]))

print("indeces : ",index+1 ,"path " ,path)








train_generator = data.flow_from_dataframe(
    df,
    "data/",
    x_col='file',
    y_col='class',
    target_size=image_size,
    class_mode='binary',
    batch_size=batch_size
)

model.fit(train_generator, epochs=3)