// 產生指定長度的陣列
function makeArray(x) {
    return (new Array(x)).fill(0);
}

// 產生指定長寬的矩陣，用來作為神經層之間的權重
function makeMatrix(x, y) {
    let grid = makeArray(x);
    return grid.map(_ => makeArray(y).map(rand));
}

// 產生 -0.5 至 0.5 之間的隨機小數
function rand() {
    return Math.random() - 0.5;
}

// 激活函式
function sigmoid(x) {
    return Math.tanh(x);
}

function relu(x) {
    return x > 0 ? x: 0;
}

// 激活函式導數
function dsigmoid(x) {
    return 1 - x * x;
}

function drelu(x) {
    return x > 0 ? 1: 0;
}

function dot(M, x) {
    if (M[0].length !== x.length) {
        throw new Error(`Dimension M(${[M.length, M[0].length]}) not fit x(${x.length})`);
    }
    return M.map(mi => mi.map((mij, j) => mij * x[j]).reduce((a, b)=>a+b));
}

function add(a, b) {
    if (a.length !== b.length) {
        throw new Error(`Dimension a(${a.length}) not fit b(${b.length})`);
    }
    if (typeof a[0] == 'object' && typeof b[0] == 'object') {
        // recursive add for multidimension vector
        return a.map((ai, i) => add(ai, b[i]));
    }
    return a.map((ai, i) => ai + b[i]);
}

function scale(a, X) {
    // multiply all the element in X by a
    if (typeof a !== 'number') {
        throw new Error(`a: ${a} is not a NUMBER`);
    }
    if (typeof X !== 'object' || X.length === undefined) {
        throw new Error(`X: ${X} is not an ARRAY`);
    }

    if (typeof X[0] == 'object') {
        return X.map(Xi => scale(a, Xi));
    }

    return X.map(Xi => a * Xi);
}

function mul(a, b) {
    // Multiply arguments element-wise
    return a.map(ai => scale(ai, b));
}



function transpose(mx) {
    return mx[0].map((_, i) => mx.map(row => row[i]));
}

function initPars(layers) {
    return {
        w: layers
        .slice(0, layers.length-1)
        .map((e, i) => [layers[i+1], e])
        .map(l => Array.from({
            length: l[0]})
            .map(_ => Array.from({
                length: l[1]})
                .map(_ => Math.random() * 2 - 1)
            )
        ),
        b: layers.slice(1).map(e => Array.from({
            length: e
        }).map(_ => Math.random()))
    };
}
function updateWeight() {}

function error(output, target) {}



// ------------------------------------------------------------- //

// nn builder
function NN(sizes, act) {
    // sizes: a list of num of neurons of each layer
    let num_layers = sizes.length;
    let {
        w, b
    } = initPars(sizes);

    const forward = (x, n = 0) => {
        if (n === w.length) return x;
        return forward(add(dot(w[n], x), b[n]).map(act), n + 1);
    };

    // const forward2 = (input) => {
    //     let tmp = input;
    //     for (let i = 0; i < w.length; i ++) {
    //         tmp = add(dot(w[i], tmp), b[i]).map(sigmoid);
    //     }
    //     // console.log(tmp);
    //     return tmp;
    // };

    function sgd(training_data, epochs, mini_batch_size, eta, test_data = null) {
        // train the neural network using mini-batch stochastic gradient descent
        // training data is a list of input and target object
        // {input:[...], target:[...]}
        if (test_data !== null) {
            let n_test = test_data.length;
        }
        let n = training_data.length;
        const shuffle = arr => arr.sort(() => Math.random() - 0.5);
        for (let i = 0; i < epochs; i++) {
            let shuffled_data = shuffle(training_data);
            let mini_batches = [];
            let j = 0;
            while (j < n) {
                mini_batches.push(shuffled_data.slice(j, j+mini_batch_size));
                j += mini_batch_size;
            }
            for (const mini_batch of mini_batches) {
                update_mini_batch(mini_batch, eta);
            }

            if (test_data !== null) {
                addLog(`Epoch ${i}: ${evaluate(test_data)} / ${n_test}`);
            } else {
                addLog(`Epoch ${i} complete`);
            }
        }

    }

    function update_mini_batch(mini_batch, eta) {
        // Update the network's weights and biases by applying
        // gradient descent using backpropagation to a single
        // mini batch.
        // mini_batch: a list of input and target object
        // eta: learning rate
        let nabla_w = w.map(wi => makeMatrix(wi.length, wi[0].length));
        let nabla_b = b.map(bi => makeArray(bi.length));
        for (const datus of mini_batch) {
            let {
                dnb, dnw
            } = backprop(datus);
            // console.log("dnb: ", dnb);
            // console.log("dnw: ", dnw);
            nabla_b = add(nabla_b, dnb);
            nabla_w = add(nabla_w, dnw);
        }
        b = add(b, scale(eta / mini_batch.length, nabla_b));
        w = add(w, scale(eta / mini_batch.length, nabla_w));
    }

    function backprop(datus) {
        // return an object `{dnb, dnw}`` representing the
        // gradient for the cost function C_x
        let {
            input, target
        } = datus;

        let nabla_w = w.map(wi => makeMatrix(wi.length, wi[0].length));
        let nabla_b = b.map(bi => makeArray(bi.length));

        // feedforward
        let activation = input;
        let activations = [input]; // list to store all the activations, layer by layer
        let zs = []; // list to store all the z vector, layer by layer
        const feedforward = (x, n = 0) => {
            if (n === w.length) return x;
            z = add(dot(w[n], x), b[n]);
            activation = z.map(sigmoid);
            zs.push(z);
            activations.push(activation);
            return feedforward(activation, n + 1);
        };
        let result = feedforward(input);

        // backward pass
        let delta = mul(
            activations.at(-1).map((ai, i) => ai - target[i]),
            zs.at(-1).map(dsigmoid)
        );

        nabla_b[nabla_b.length-1] = delta;
        nabla_w[nabla_w.length-1] = mul(delta, activations.at(-2));
        for (let l = 2; l < num_layers; l++) {
            let z = zs.at(-l);
            let ds = z.map(dsigmoid);
            delta = mul(dot(transpose(w.at(-l+1)), delta), ds);
            nabla_b[nabla_b.length-l] = delta;
            nabla_w[nabla_b.length-l] = mul(delta, activations.at(-l-1));
        }
        return {
            dnb: nabla_b, dnw: nabla_w
        };
    }

    function evaluate() {}

    const get_weight = () => w;
    const get_bias = () => b;

    return {
        forward, backprop, update_mini_batch, get_weight, get_bias, sgd
    };
}

// function NN(a, b, c, d) {

//     let n1 = makeArray(a); //第 1 層神經儲存的數值
//     let n2 = makeArray(b); //第 2 層神經儲存的數值
//     let n3 = makeArray(c); //第 3 層神經儲存的數值
//     let n4 = makeArray(d); //第 4 層神經儲存的數值
//     let w1 = makeMatrix(a, b); //第 1, 2 層神經之間的權重
//     let w2 = makeMatrix(b, c); //第 2, 3 層神經之間的權重
//     let w3 = makeMatrix(c, d); //第 3, 4 層神經之間的權重

//     // 正向傳遞
//     function forward(inputs) {
//         n1 = inputs;
//         n2 = dot(n1, w1).map(sigmoid);
//         n3 = dot(n2, w2).map(sigmoid);
//         n4 = dot(n3, w3).map(sigmoid);
//         return n4
//     }

//     // 反向傳遞
//     function backward(target, learningRate = 0.001) {
//         let e4 = target.map((t, i) => t - n4[i]); // actually this is d-cost function
//         let d4 = n4.map(dsigmoid).map((v, i) => v * e4[i]);
//         let e3 = dot(d4, transpose(w3));
//         let d3 = n3.map(dsigmoid).map((v, i) => v * e3[i]);
//         let e2 = dot(d3, transpose(w2));
//         let d2 = n2.map(dsigmoid).map((v, i) => v * e2[i]);
//         updateWeight(w3, n3, d4, learningRate);
//         updateWeight(w2, n2, d3, learningRate);
//         updateWeight(w1, n1, d2, learningRate);
//     }

//     return {
//         forward, backward
//     }
// }

// ------------------------------------------------------------- //

let training = false; //是否正在訓練
var trainData = []; //訓練資料
let testData = null; //測試資料

let LABEL = 10; //讀取資料的高度
let LENGTH = 1000; //讀取資料的長度
let CUT = 90; //切割資料長度，區分訓練和測試
const LAYERS = [28 * 28, 16, 10];
loadImage('mnist.jpg');

// let pars = initPars(LAYERS);
// var nn = NN(pars);
var nn = NN(LAYERS, sigmoid);

function detect() {
    let input = getHandDrawnData();
    let forward = nn.forward(input);
    addLog(`Detect: ${forward.indexOf(Math.max(...forward))}`);
}

// 點擊「開始訓練」按鈕觸發此程式
function start() {
    training = true;
    trainData = Array.from({
        length: LENGTH
    }).map((_, x) =>
        // target using one-hot encoding
        Array.from({
            length: LABEL
        }).map((_, y) => {
            return {
                input: getImageData(x * 28, y * 28, 28, 28),
                target: Array.from({
                    length: LABEL
                }).map((_, i) => i == y ? 1: 0), // one-hot encoding target
            };
        })).flat();
    // testData = []; //清除測試資料


    loop(); // 開始訓練迴圈
}

// 點擊「暫停訓練」按鈕觸發此程式
function stop() {
    training = false;
    console.log(nn.get_bias())
}

// 訓練的迴圈
function loop(round = 0) {
    // markCtx.clearRect(0, 0, 28000, 420); //清除標記

    let trainPass = train([trainData[round % trainData.length]], 0.01); //訓練
    let trainPct = (trainPass.count * 100 / trainData.length).toFixed(2); //計算訓練通過率
    addLog(`Epoch: ${round}, Accuracy: ${trainPct} %, cost: ${trainPass.cost.toFixed(3)}`);
    if (training) setTimeout(() => loop(round + 1), 100);
}

function dist2(a, b) {
    return a.map((e, i) => Math.pow(e-b[i], 2)).reduce((a, b) => a+b);
}

// quadratic cost function <- this is what we try to minimize
// C(w,b) = 1/2n Sum_x (y(x)-a)^2
function evaluate(output, target) {
    return output
    .map((e, i) => dist2(e, target[i]))
    .reduce((a, b) => a+b) / output.length / 2;
}

// function getRandomSubarray(arr, size) {
//     let shuffled = arr.slice(0), i = arr.length, temp, index;
//     while (i--) {
//         index = Math.floor((i + 1) * Math.random());
//         temp = shuffled[index];
//         shuffled[index] = shuffled[i];
//         shuffled[i] = temp;
//     }
//     return shuffled.slice(0, size);
// }


// 執行訓練資料
function train(data, eta) {
    const target = data.map(e => e.target);

    // training
    nn.update_mini_batch(data, eta);

    const forward = data.map(e => {
        let output = nn.forward(e.input);
        let result = output.indexOf(Math.max(...output));
        let target = e.target.indexOf(1);
        return {
            result, target, output
        };
    });

    const output = forward.map(e => e.output);
    let count = forward.filter(e => e.result == e.target).length;
    let cost = evaluate(output, target);

    return {
        count, cost
    };
}

// 執行測試資料
function test(data) {}