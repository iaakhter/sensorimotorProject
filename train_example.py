import model_example
import tensorflow as tf
import processImages
import numpy as np

class trainCNN:
    def __init__(self):
        # Load Training Data
        imagePath ="trainingData/trainingFeature.txt"
        labelPath = "trainingData/trainingLabel.txt"
        self.numOfExamples = 2000

        traningImagesPath = "trainingData/resizedImages/image"

        #To train using centers, uncomment next two lines
        xTrain = processImages.constructXFromTargetFocusLocations(self.numOfExamples, imagePath)
        self.xTrain = np.reshape(xTrain, (self.numOfExamples, 1, 2, 1))

        #To train using the 50x50 image in grey, uncomment the next two lines
        # xTrain = processImages.convertImageToArray(self.numOfExamples, traningImagesPath)
        # self.xTrain = np.reshape(xTrain, (self.numOfExamples, 50, 50, 1))

        yTrain = processImages.convertLabelToArray(self.numOfExamples, labelPath)
        self.yTrain, self.muYTrain, self.stdYTrain = processImages.standardizeCols(yTrain)

    def train(self):
        #batch variables
        epochs_completed = 0
        index_in_epoch = 0
        # Start session
        sess = tf.InteractiveSession()
        # Learning Functions
        L2NormConst = 0.001
        train_vars = tf.trainable_variables()
        loss = tf.reduce_mean(tf.square(tf.subtract(model_example.y_, model_example.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

        sess.run(tf.global_variables_initializer())

        # Training loop variables
        epochs = 200
        batch_size = 100
        num_samples = self.numOfExamples
        step_size = int(num_samples / batch_size)

        for epoch in range(epochs):
            for i in range(step_size):
                start = index_in_epoch
                index_in_epoch += batch_size
                if index_in_epoch >= self.numOfExamples:
                    #Finished epoch
                    epochs_completed += 1
                    #Start new epoch
                    start = 0
                    index_in_epoch = batch_size
                end = index_in_epoch
                batch = self.xTrain[start:end], self.yTrain[start:end]

                train_step.run(feed_dict={model_example.x: batch[0], model_example.y_: batch[1], model_example.keep_prob: 0.8})

                if i%10 == 0:
                  loss_value = loss.eval(feed_dict={model_example.x:batch[0], model_example.y_: batch[1], model_example.keep_prob: 1.0})
                  print("epoch: %d step: %d loss: %g"%(epoch, epoch * batch_size + i, loss_value))

        # Save the Model
        saver = tf.train.Saver()
        saver.save(sess, "model.ckpt")

if __name__ == "__main__":
    modelTest = trainCNN()
    modelTest.train()
