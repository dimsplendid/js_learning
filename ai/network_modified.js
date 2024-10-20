var nj = require('numjs')

function init_pars(layers) {
    return {
        w: layers
            .slice(0, layers.length-1)
            .map((e,i) => [layers[i+1], e])
            .map(wi => nj.random(wi[0], wi[1]))

    }
}

console.log(nj.random([1,2,3]))