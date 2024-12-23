// import { Sequential } from "@tensorflow/tfjs";
import { MnistData } from "./tf_cnn_data.js";

// global variable for type checking
/** @type {import("@tensorflow/tfjs")} */
const tf = window.tf;
/** @type {import("@tensorflow/tfjs-vis")} */
const tfvis = window.tfvis;

/**
 * Show some example of MNIST Data
 * @param {MnistData} data 
 */
async function showExample(data) {
    // Create a container in the visor
    const surface = tfvis.visor().surface({
        name: 'Input Data Examples',
        tab: 'Input Data'
    });

    // Get the examples
    const examples = data.nextTestBatch(30);
    const numExamples = examples.xs.shape[0];

    // Create a canvas element to render each example
    for (let i = 0; i < numExamples; i++) {
        const imageTensor = tf.tidy(() => {
            // Reshape the image to 28x28 px
            return examples.xs
                .slice([i,0], [1, examples.xs.shape[1]])
                .reshape([28, 28, 1]);
        });

        const canvas = document.createElement('canvas',{ willReadFrequently: true } );
        // let ctx = canvas.getContext('2d', {willReadFrequently: true});
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = 'margin: 4px;';
        await tf.browser.toPixels(imageTensor, canvas);
        surface.drawArea.appendChild(canvas);

        imageTensor.dispose();
    }
}

function getModel() {
    const model = tf.sequential();

    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1; // the fancy way to say color :D

    // In the first layer of our convolutional neural network we have
    // to specify the input shape. Then we specify some parameters for
    // the convolution operation that takes place in this layer.
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }))

    // The MaxPooling layer acts as a sort of downsampling using max values
    // in a region instead of averageing.
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2,2]
    }));

    // Repeat another conv2d + maxPooling stack.
    // Note thate we have more filters in the convolution
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }))

    // The MaxPooling layer acts as a sort of downsampling using max values
    // in a region instead of averageing.
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2,2]
    }));

    // Now we flatten the output form the 2Dfilters into a 1D vector to prepare
    // higher dimensional data to a final classification output layer
    model.add(tf.layers.flatten())

    // Our last layer is a dense layer which has 10 outpout units, one for each
    // output class (i.e. 0, 1, 2, ..., 9)
    const NUM_OUTPUT_CLASSES = 10;
    model.add(tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
    }));
    
    // Choose an optimizer, loss function and accuracy metric,
    // then compile and return the model
    const optimizer = tf.train.adam();
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    })

    return model;
}

/**
 * 
 * @param {Sequential} model 
 * @param {MnistData} data 
 */
async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
        name: 'Model Training', tab: 'Training', style: {height: '1000px'}
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;

    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
        return [
            d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });

    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        return [
            d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });

    return model.fit(
        trainXs, trainYs, {
            batchSize: BATCH_SIZE,
            validationData: [testXs, testYs],
            epochs: 10,
            shuffle: true,
            callbacks: fitCallbacks
        }
    );
}

async function run() {
    const data = new MnistData();
    await data.load();
    await showExample(data);

    const model = getModel();
    tfvis.show.modelSummary({name: 'Model Architecture', tab: 'model'}, model);

    await train(model, data);
}

document.addEventListener('DOMContentLoaded', run)