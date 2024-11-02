/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getDate() {
    const cars_data_res = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const cars_data = await cars_data_res.json();
    const cleaned = cars_data.map(car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
    }))
    .filter(car => (car.mpg != null && car.horsepower != null));

    return cleaned;
}

async function run() {
    // Load and plot the original input data that we are going to train on.
    const data = await getDate();
    const values = data.map(d => ({
        x: d.horsepower,
        y: d.mpg,
    }));

    tfvis.render.scatterplot(
        {name: 'Horsepower v MPG'},
        {values},
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
        }
    )
    
    // TODO:
}

document.addEventListener('DOMContentLoaded', run)