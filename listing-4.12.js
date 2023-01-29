// Building a forEach function to demonstrate setting a function's context
function forEach(list, callback) {
    for (var n = 0; n < list.length; n++)
        callback.call(list[n], n);
}

// Create an array of objects
var weapons = [{type: 'shuriken'}, {type: 'katana'}, {type: 'nunchucks'}];

forEach(weapons, function(index) {
    console.log('Weapon ' + index + ': ' + this.type);
});