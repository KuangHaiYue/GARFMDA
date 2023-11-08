import numpy as np
from keras.applications.densenet import DenseNet121
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
def train44(x_train,y_train,x_test,y_test):
 base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
 for layer in base_model.layers:
    layer.trainable = False
# 添加全局平均池化层
 x = base_model.output
 x = GlobalAveragePooling2D()(x)
# 添加全连接层
 predictions = Dense(2, activation='softmax')(x)
# 构建模型
 model = Model(inputs=base_model.input, outputs=predictions)
 model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 训练模型
 model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test))
 y_pred = model.predict(x_test)
 return y_pred, y_test