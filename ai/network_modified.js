const SIZE = 28
var dataCanvas = $('#data-canvas')[0]
var dataCtx = dataCanvas.getContext('2d', { willReadFrequently: true })
var drawCanvas = $('#draw-canvas')[0]
var drawCtx = drawCanvas.getContext('2d', { willReadFrequently: true })
var markCanvas = $('#mark-canvas')[0]
var markCtx = markCanvas.getContext('2d', { willReadFrequently: true })
var miniCanvas = $('#mini-canvas')[0]
var miniCtx = miniCanvas.getContext('2d', { willReadFrequently: true })

var cursorX = 0
var cursorY = 0
var isMouseDown = false

initDataCanvas();
loadCache();

function initDataCanvas() {
    dataCtx.fillStyle = 'back'
    dataCtx.fillRect(0, 0, 28000, 280);
}

function clearDataCanvas() {
    if (confirm('確定要清除？')) {
        initDataCanvas();
        localStorage.setItem('image', '');
    }
}

function loadCache() {
    var dataURL = localStorage.getItem('image')
    var img = new Image
    img.src = dataURL
    img.onload = () => dataCtx.drawImage(img, 0, 0)
}

function loadImage(src) {
    var img = new Image
    img.onload = () => dataCtx.drawImage(img, 0, 0)
    img.src = src
}

function download() {
    var link = document.createElement('a')
    link.download = 'backup.png';
    link.href = dataCanvas.toDataURL('image/png')
    link.click();
}

function upload() {
    const fileInput = document.createElement('input')
    fileInput.type = 'file'
    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0]
        const reader = new FileReader()
        reader.onload = (e) => {
            const img = new Image()
            img.onload = () => dataCtx.drawImage(img, 0, 0)
            img.src = e.target.result
        }
        reader.readAsDataURL(file)
    })
    fileInput.click()
}

function handleFileSelect() {
    const file = event.target.files[0]
}

var chart = new Chart($('#chart-canvas')[0], {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            { label: '訓練資料通過率', borderWidth: 2 },
            { label: '測試資料通過率', borderWidth: 2 },
        ]
    },
    options: {
        elements: { point: { radius: 0 } },
        scales: { y: { beginAtZero: true, min: 0, max: 100 } },
        animation: { duration: 0 },
    }
})

var isTraining = false
$('#start').click(() => {
    $('#start').toggleClass('btn-success')
    $('#start').toggleClass('btn-danger')
    $('#start').text(isTraining ? '開始訓練' : '停止訓練')
    isTraining ? stop() : start()
    isTraining = !isTraining
})
$('#stop').click(() => window.stop())
$('#detect').click(() => window.detect())
$('#saveback').click(() => window.saveback())

$('#next').click(() => {
    saveback()
    cursorX += 1
    loadFromData()
    $('#cursor').css({ top: `${cursorY * SIZE}px`, left: `${cursorX * SIZE}px` })
})

$('#draw-canvas').on('mousedown', () => isMouseDown = true)
$('#draw-canvas').on('mouseup', () => isMouseDown = false)
$('#draw-canvas').on('mousemove', (e) => {
    if (!isMouseDown) return
    var rect = drawCanvas.getBoundingClientRect()
    var x = Math.floor((e.clientX - rect.left) / 10)
    var y = Math.floor((e.clientY - rect.top) / 10)
    drawCtx.fillStyle = 'white'
    drawCtx.beginPath()
    drawCtx.arc(x, y, 1, 0, 3 * Math.PI)
    drawCtx.fill()
})
$('#mark-canvas').click(e => {
    var rect = dataCanvas.getBoundingClientRect()
    cursorX = Math.floor((e.clientX - rect.left) / SIZE)
    cursorY = Math.floor((e.clientY - rect.top) / SIZE)
    loadFromData()
    $('#cursor').css({ top: `${cursorY * SIZE}px`, left: `${cursorX * SIZE}px` })
})
$('#upload').change(e => {
    var img = new Image()
    img.onload = () => dataCtx.drawImage(img, 0, 0)
    img.src = URL.createObjectURL(e.target.files[0])
})
$('#clear').click(() => {
    drawCtx.fillStyle = 'black'
    drawCtx.fillRect(0, 0, SIZE, SIZE)
})

function loadFromData() {
    drawCtx.clearRect(0, 0, SIZE, SIZE)
    drawCtx.drawImage(dataCanvas, cursorX * SIZE, cursorY * SIZE, SIZE, SIZE, 0, 0, SIZE, SIZE)
}

function saveback() {
    dataCtx.drawImage(drawCanvas, 0, 0, SIZE, SIZE, cursorX * SIZE, cursorY * SIZE, SIZE, SIZE)
    var data = dataCanvas.toDataURL('image/jpg')
    localStorage.setItem('image', data)
}

function getHandDrawnData() {
    var arr = []
    miniCtx.drawImage(drawCanvas, 0, 0, SIZE, SIZE)
    var data = miniCtx.getImageData(0, 0, SIZE, SIZE).data
    for (var i = 0; i < data.length; i += 4) {
        ;
        var avg = (data[i + 0] + data[i + 1] + data[i + 2]) / 3 / 225
        arr.push(avg)
    }
    return arr
}

function getImageData(x, y, width, height) {
    var list = []
    var data = dataCtx.getImageData(x, y, width, height).data
    for (var i = 0; i < data.length; i += 4) {
        var avg = (data[i + 0] + data[i + 1] + data[i + 2]) / 3 / 225
        list.push(avg)
    }
    return list
}

function addLog(content) {
    $('#logger').prepend(`<span>${JSON.stringify(content)}</span><br/>`)
}

function chartLog(trainRate, testRate) {
    chart.data.labels.push(chart.data.labels.length + 1)
    chart.data.datasets[0].data.push(trainRate)
    chart.data.datasets[1].data.push(testRate)
    chart.update()
}
// var nj = require('numjs');

/**
 * Initialize a neural networks weights and bias with random number.
 * @param {number[]} layers - The number of neuron for each layer(include input)
 * @returns {{w: NdArray[], b: NdArray[]}} 
 *      The neural networks constain weights(w) and bias(b)
 *      The initial weights and bias is random from -0.5 to +0.5
 * @example
 * // initialize a neural network
 * const nn = nn_init([3,2,1]);
 * // nn = {w:[...], b:[...]}
 */
function nn_init(layers) {
    const weights = layers.slice(0, -1)
        .map((_, i) => nj.random(layers[i + 1], layers[i]).subtract(0.5));
    const biases = layers.slice(1).map(b => nj.random(b, 1).subtract(0.5));
    return { w: weights, b: biases }
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
function nn_forward(nn, input, log = false) {

    const forward = (nn, input, log_data = null, layer = 0) => {
        if (layer === nn.w.length) return [input, log_data];

        const z = nn.w[layer].dot(input).add(nn.b[layer]);
        const a = nj.sigmoid(z);

        const update_log = log_data ? {
            zs: [...log_data.zs, z],
            as: [...log_data.as, a]
        } : null;

        return forward(nn, a, update_log, layer + 1)
    }

    const initial_log = log ? { zs: [], as: [input] } : null;
    const [predict, final_log] = forward(nn, input, initial_log)
    return { predict, log: final_log }
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
 * representing the gradient for the cost function C_x.  `nabla_b`
 * and`nabla_w` are layer-by-layer lists of NdArrays,
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
        delta = nn.w.at(-l + 1).T.dot(delta).multiply(sp);
        nabla_w = nabla_w.with(-l, delta.dot(as.at(-l - 1).T));
        nabla_b = nabla_b.with(-l, delta);
    }

    return { w: nabla_w, b: nabla_b };
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

    return { w: updated_weights, b: updated_biases };
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

function shuffle(array) {
    let tmp, current, top = array.length;

    if(top) while(--top) {
        current = Math.floor(Math.random() * (top + 1));
        tmp = array[current];
        array[current] = array[top];
        array[top] = tmp;
    }

    return array;
}

// Main

let training = false; //是否正在訓練
let trainData = []; //訓練資料
let testData = null; //測試資料
let learning_rate = 0.05;

const LABEL = 10; //讀取資料的高度 -> 種類
const LENGTH = 1000; //讀取資料的長度 -> 每種的資料數量
const TEST_RATIO = 0.2; //切割資料長度，區分訓練和測試
// const LAYERS = [28 * 28, 20, LABEL];
const LAYERS = [28 * 28, 32, 16, LABEL];

// loadImage('mnist.jpg');
loadImage('mnist_fashion.jpg');

let nn = nn_init(LAYERS);

function detect() {
    const input = nj.array(getHandDrawnData()).reshape([28*28, 1]);
    const {predict} = nn_forward(nn, input);
    const predict_list = predict.flatten().tolist();
    console.log(input.flatten().tolist().slice(0,10));
    console.log(predict);
    addLog(`Detect: ${predict_list.indexOf(predict.max())}`);
}

function start() {
    training = true;

    all_data = Array.from({length: LENGTH * LABEL})
        .map((_, i) => {
            const x = Math.floor(i / LABEL);
            const y = i % LABEL;
            const input = nj.array(getImageData(x * 28, y * 28, 28, 28))
                .reshape([28 * 28, 1]);
            let target = nj.zeros([LABEL, 1]);
            target.set(y,0,1);
            return {input, target, x, y};
        });
    let cutting = Math.floor(LENGTH * LABEL * (1-TEST_RATIO));

    // shuffle(all_data);
    trainData = all_data.slice(0, cutting);
    testData = all_data.slice(cutting);
    loop();
}

function stop() {
    training = false;
}

function loop(round=1, total_time=0) {
    markCtx.clearRect(0, 0, 28000, 420); //清除標記

    const start_time = Date.now();

    const train_accuracy = (train(trainData, learning_rate) * 100).toFixed(2);
    const test_accuracy = (test(testData) * 100).toFixed(2)

    const duration = (Date.now() - start_time) / 1000;
    const avg_duration = (duration + total_time) / round;

    addLog(`Train: ${train_accuracy}, Test: ${test_accuracy} %, Training Time: ${duration.toFixed(2)} sec, avg Time: ${avg_duration.toFixed(2)} sec`);
    chartLog(train_accuracy, test_accuracy);

    if (training) setTimeout(loop, 100, round+1, total_time + duration);
}

/**
 * training neural network and return the accuracy
 * @param {{input: NdArray[], target: NdArray[], x: number, y: number}[]} data 
 *      - The Training data, (x, y) is the figure
 *        position in canvas
 * @param {number} eta - learning rate
 * @returns The accuracy of the prediction
 */
function train(data, eta) {

    const accuracy = data.map(({input, target, x, y}) => {

        const nabla = nn_backprop(nn,input, target);
        nn = nn_update(nn, nabla, eta); // side-effect: update global nn
        
        const {predict} = nn_forward(nn, input);
        const predict_number = predict.flatten().tolist().indexOf(predict.max());
        
        const is_pass = target.get(predict_number,0);
        
        markCtx.fillStyle = is_pass ? 'blue' : 'red';
        markCtx.fillRect(x * SIZE, y * SIZE, SIZE, SIZE); // side-effect: draw UI
        
        return is_pass === 1 ? 1 : 0;
    }).reduce((a,b) => a + b) / data.length;
    
    return accuracy;
}

/**
 * test neural network and return the accuracy
 * @param {{input: NdArray[], target: NdArray[], x: number, y: number}[]} data
 *       - The Test data, (x, y) is the figure
 *        position in canvas
 * @returns The accuracy of the prediction
 */
function test(data) {

    const accuracy = data.map(({input, target, x, y}) => {
        
        const {predict} = nn_forward(nn, input);
        const predict_number = predict.flatten().tolist().indexOf(predict.max());
        
        const is_pass = target.get(predict_number,0);
        
        markCtx.fillStyle = is_pass ? 'green' : 'red';
        markCtx.fillRect(x * SIZE, y * SIZE, SIZE, SIZE); // side-effect: draw UI
        
        return is_pass === 1 ? 1 : 0;
    }).reduce((a,b) => a + b) / data.length;
    
    return accuracy;
}