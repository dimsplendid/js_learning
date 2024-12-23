
// global variable for type checking
/** @type {import("@tensorflow/tfjs")} */
const tf = window.tf;
/** @type {import("@tensorflow/tfjs-vis")} */
const tfvis = window.tfvis;


/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
    const carsDataResponse = await fetch('./carsData.json');
    const carsData = await carsDataResponse.json();
    const cleaned = carsData.map(car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
    }))
        .filter(car => (car.mpg != null && car.horsepower != null));

    return cleaned;
}

function createModel() {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single input layer
    model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

    // hidden layer
    model.add(tf.layers.dense({units: 16, activation: 'relu'}));

    // Add an output layer
    model.add(tf.layers.dense({ units: 1, useBias: true }));

    return model;
}

/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important bast practices fo _shuffling_
 * the data adn _normalizing_ the data
 * MPG on the y-axis
 */
function convertToTensor(data) {
    // Wrapping these calculations in a tidy will dispose any
    // inter mediate tnesors.

    return tf.tidy(() => {
        // Step 1. Shuffle the data
        tf.util.shuffle(data);

        // Step 2. Convert data to Tensor
        const inputs = data.map(d => d.horsepower);
        const labels = data.map(d => d.mpg);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        // Step 3. Normalize the data to the range 0 - 1 using min-max scaling
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            // Return the min/max bounds so we can use them later
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        }
    });
}

async function trainModel(model, inputs, labels) {
    // Prepare themodel for training.
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        matrics: ['mse']
    })

    const batchSize = 64;
    const epochs = 256;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance' },
            ['loss', 'mse'],
            { height: 200, callbacks: ['onEpochEnd'] }
        )
    })
}

function testModel(model, inputData, normalizationData) {
    const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

    // Gernerate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing he inverse of the min-max scaling
    // that we did earlier.

    const [xs, preds] = tf.tidy(() => {

        const xsNorm = tf.linspace(0, 1, 100);
        const predictions = model.predict(xsNorm.reshape([100, 1]));

        // Un-normalize the data
        const unNormXs = xsNorm
            .mul(inputMax.sub(inputMin))
            .add(inputMin);

        const unNormPreds = predictions
            .mul(labelMax.sub(labelMin))
            .add(labelMin);

        return [unNormXs.dataSync(), unNormPreds.dataSync()]
    })

    const predictPoints = Array.from(xs).map((val, i) => {
        return { x: val, y: preds[i] }
    });

    const originalPoints = inputData.map(d => ({
        x: d.horsepower, y: d.mpg,
    }));

    tfvis.render.scatterplot(
        { name: 'Model Predictions vs Original Data' },
        { 
            values: [originalPoints, predictPoints], 
            series: ['origianl', 'predicted'] 
        },
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300,
        }
    )
}

async function run() {
    // Load and plot the original input data that we are going to train on.
    const data = await getData();
    const values = data.map(d => ({
        x: d.horsepower,
        y: d.mpg,
    }));

    tfvis.render.scatterplot(
        { name: 'Horsepower v MPG' },
        { values },
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
        }
    );

    // More code will be added below

    const model = createModel();
    tfvis.show.modelSummary({ name: 'Model Summary' }, model)

    // Convert the data to a form we can use for training
    const tensorData = convertToTensor(data);
    const { inputs, labels } = tensorData;

    // Train the model
    await trainModel(model, inputs, labels);
    console.log('Done Training')

    // Test the model
    testModel(model, data, tensorData);
}

document.addEventListener('DOMContentLoaded', run);