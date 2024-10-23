var nj = require('numjs');

/**
 * Initialize a neural networks weights and bias with random number.
 * @param {number[]} layers - The number of neuron for each layer(include input)
 * @returns {{w: NdArray[], b: NdArray[]}} The neural networks constain weights(w) and bias(b)
 * @example
 * // initialize a neural network
 * const nn = nn_init([3,2,1]);
 * // nn = {w:[...], b:[...]}
 */
function nn_init(layers) {
    const weights = layers.slice(0,-1)
        .map((_,i) => nj.random(layers[i+1], layers[i]));
    const biases = layers.slice(1).map(b => nj.random(b, 1));
    return {w: weights, b: biases}
}

/**
 * 
 * @param {{w: NdArray[], b: NdArray[]}} nn - neural network
 * @param {NdArray} input - shape(N, 1) ndarray, N should be
 *                           the same with nn.w[layer] size.
 * @param {boolean} log - defualt is `false`, whether to log result and 
 *                          activation for each layer
 * @returns {object} the {predict, log} of nn
 * 
 * @example
 * const input = nj.array([[3,2,1]])
 * let {predict, log} = nn_forward(nn, input, log=true)
 * // or let {predict} = nn_forward(nn, input)
 */
function nn_forward(nn, input, log=false) {  

    const forward = (nn, input, log_data=null, layer=0) => {
        if (layer === nn.w.length) return [input, log_data];

        const z = nn.w[layer].dot(input).add(nn.b[layer]);
        const a = nj.sigmoid(z);

        const update_log = log_data ? {
            zs: [...log_data.zs, z],
            as: [...log_data.as, a]
        } : null;

        return forward(nn, a, update_log, layer+1)
    }

    const initial_log = log ? {zs: [], as: [input]} : null;
    const [predict, final_log] = forward(nn, input, initial_log)
    return {predict, log: final_log}
}

/**
 * Return the derivative of sigmoid function of input array, element-wise.
 * @param {Array|NdArray|number} x 
 * @returns {NdArray}
 */
function dsigmoid(x) {
    return nj.sigmoid(x).multiply(nj.ones(x.shape).subtract(nj.sigmoid(x)));
}

/**
 * Calculate nabla of neural network using backward propagation algorithm
 * @param {{w: NdArray[], b: NdArray[]}} nn - neural network
 * @param {NdArray} input - shape(N, 1) ndarray, N should be
 *                           the same with nn.w[layer] size.
 * @param {NdArray} target - shape(M, 1) ndarray, the one-hot
 *                           encoding result
 * @returns {{w: NdArray[], b: NdArray[]}} the nabla w and b of the neural network
 * 
 * @example
 * let nabla = nn_backprop(nn, input, target);
 * nn = nn_update(nn, nabla, eta=0.01);
 */
function nn_backprop(nn, input, target) {
    let nabla_w = nn.w.map(wi => nj.zeros(wi.shape));
    let nabla_b = nn.b.map(bi => nj.zeros(bi.shape));

    const { log } = nn_forward(nn, input, true);
    const { zs, as } = log;
    let delta = nj.multiply(as.at(-1).subtract(target), dsigmoid(zs.at(-1)));

    nabla_w = nabla_w.with(-1, delta.dot(as.at(-2).T));
    nabla_b = nabla_b.with(-1, delta);

    for (let l = 2; l <= nn.w.length; l++) {
        let z = zs.at(-l);
        let sp = dsigmoid(z);
        delta = nn.w.at(-l+1).T.dot(delta).multiply(sp);
        nabla_w = nabla_w.with(-l, delta.dot(as.at(-l-1).T));
        nabla_b = nabla_b.with(-l, delta);
    }

    return {w: nabla_w, b: nabla_b};
}

/**
 * Update neural network's weights by nabla
 * @param {{w: NdArray[], b: NdArray[]}} nn - neural network
 * @param {{w: NdArray[], b: NdArray[]}} nabla - the nabla of the neural network
 * @param {number} eta - learning rate
 * @returns {{w: NdArray[], b: NdArray[]}} weight-updated neural network
 * @example
 * let nabla = nn_backprop(nn, input, target);
 * nn = nn_update(nn, nabla, eta=0.01);
 */
function nn_update(nn, nabla, eta) {
    const updated_weights = nn.w.map((wi, i) => wi.subtract(nabla.w[i].multiply(eta)));
    const updated_biases = nn.b.map((bi, i) => bi.subtract(nabla.b[i].multiply(eta)));

    return {w: updated_weights, b: updated_biases};
}

/**
 * Calculate the distance sqaure between 2 vector
 * @param {NdArray} v1 - vector 1
 * @param {NdArray} v2 - vector 2
 * @returns {number} the distance sqaure of v1 and v2
 */
function dist_square(v1, v2) {
    let delta = v1.subtract(v2);
    return nj.sum(delta.T.dot(delta));
}

// // test main
// let nn = nn_init([4,2,3]);
// let input = nj.array([1, 0, 0, 0]).reshape([4,1]);
// let target = nj.array([0,0,1]).reshape([3,1])

// // test training
// let count = 0;
// let eta = 0.1;
// while (count < 10000) {
//     if (count % 100 === 0) {
//         let {predict} = nn_forward(nn, input);
//         console.log(`The distance to target: ${dist_square(predict, target)}`)
//     }
//     nabla = nn_backprop(nn, input, target);
//     nn = nn_update(nn, nabla, eta);
//     count++;
// }

// In Orange Apple project

let training = false; //是否正在訓練
var trainData = []; //訓練資料
let testData = null; //測試資料

const LABEL = 10; //讀取資料的高度
const LENGTH = 1000; //讀取資料的長度
const CUT = 900; //切割資料長度，區分訓練和測試
const LAYERS = [28 * 28, 16, 10];

loadImage('mnist.jpg');

var nn = nn_init(LAYERS);

function detect() {
    const input = nj.array(getHandDrawnData()).reshape([28*28,1]);
    const {predict} = nn_forward(nn, input);
    const predict_list = predict.flatten().tolist();
    addLog(`Detect: ${predict_list.indexOf(predict.max())}`);
}

function start() {
    training = true;

    trainData = Array.from({length: LENGTH * LABEL})
        .map((_, i) => {
            const x = Math.floor(i / LABEL);
            const y = i % LABEL;
            const input = nj.array(getImageData(x * 28, y * 28, 28, 28))
                .reshape([28 * 28, 1]);
            let target = nj.zeros([LABEL, 1]);
            target.set(y,0,1);
            return {input, target};
        });
    loop();
}

function stop() {
    training = false;
}

function loop(round=0) {
    markCtx.clearRect(0, 0, 28000, 420); //清除標記

    let count = train(trainData, 0.01);
    let accuracy = (count / trainData.length * 100).toFixed(2);
    addLog(`Round-${round}: Accuracy: ${accuracy} %`);
    if (training) setTimeout(loop, 100, round+1);
}

function train(training_data, eta) {
    let count = 0;
    for (let i = 0; i < training_data.length; i++){
        const {input, target} = training_data[i];
        const nabla = nn_backprop(nn,input, target);
    
        nn = nn_update(nn, nabla, eta);
        
        const {predict} = nn_forward(nn, input);
        const predict_class = predict.flatten().tolist().indexOf(predict.max());

        if(target.get(predict_class, 0)===1) count++;
    }
    return count;
}