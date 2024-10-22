var nj = require('numjs');

/**
 * Initialize a neural networks weights and bias with random number.
 * @param {number[]} layers - The number of neuron for each layer(include input)
 * @returns {object} The object constain weights(w) and bias(b)
 * @example
 * // initialize a neural network
 * const nn = nn_init([3,2,1]);
 * // nn = {w:[...], b:[...]}
 */
function nn_init(layers) {
    return {
        w: layers
            .slice(0, layers.length-1)
            .map((e,i) => [layers[i+1], e])
            .map(wi => nj.random(wi[0], wi[1])),
        b: layers
            .slice(1)
            .map(bi => nj.random(bi, 1))
    }
}

/**
 * 
 * @param {object} nn - neural network
 * @param {NdArray} input - shape(N, 1) ndarray, N should be
 *                           the same with nn.w[layer] size.
 * @param {boolean} log - log to record result and activation for each layer
 * @returns {object} the {predict, log} of nn
 * 
 * @example
 * const input = nj.array([[3,2,1]])
 * let {predict, log} = nn_forward(nn, input)
 */
function nn_forward(nn, input, log=false) {  
    
    let tmp_log = null
    if (log) tmp_log = {zs:[], as:[input]};

    const forward = (nn, input, tmp_log=null, layer=0) => {
        if (layer === nn.w.length) return input;
        let z = nn.w[layer].dot(input).add(nn.b[layer])
        let a = nj.sigmoid(z);
        if (tmp_log !== null) {
            tmp_log.zs.push(z);
            tmp_log.as.push(a);
        }
        return forward(nn, a, tmp_log, layer+1)
    }

    let predict = forward(nn, input, tmp_log)
    return {predict: predict, log: tmp_log}
}

/**
 * Return the derivative of sigmoid function of input array, element-wise.
 * @param {Array|NdArray|number} x 
 * @returns {NdArray}
 */
function dsigmoid(x) {
    return nj.sigmoid(x).multiply(nj.ones(x.shape).subtract(nj.sigmoid(x)));
}

function nn_backprop(nn, input, target) {
    let nabla_w = nn.w.map(wi => nj.zeros(wi.shape));
    let nabla_b = nn.b.map(bi => nj.zeros(bi.shape));

    let {log} = nn_forward(nn, input, true);
    let {zs, as} = log;
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

function nn_update(nn, nabla, eta) {
    nn.w = nn.w.map((wi, i) => wi.subtract(nabla.w[i].multiply(eta)));
    nn.b = nn.b.map((bi, i) => bi.subtract(nabla.b[i].multiply(eta)));

    return nn
}

function dist_square(v1, v2) {
    let delta = v1.subtract(v2);
    return nj.sum(delta.T.dot(delta));
}
// test main
let nn = nn_init([4,2,3]);
let input = nj.array([1, 0, 0, 0]).reshape([4,1]);
let target = nj.array([0,0,1]).reshape([3,1])

// test train
let count = 0;
let eta = 0.1;
while (count < 10000) {
    if (count % 100 === 0) {
        let {predict} = nn_forward(nn, input);
        console.log(`The distance to target: ${dist_square(predict, target)}`)
    }
    nabla = nn_backprop(nn, input, target);
    nn = nn_update(nn, nabla, eta);
    count++;
}