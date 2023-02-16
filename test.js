// const cover = (i) => console.log("cover", i)

// setTimeout(() => {
//   cover(0)
// }, 1000);
// cover(1);

// const cover_wait = new Promise((resolve, reject) => {
//   setTimeout(() => {
//     cover(0)
//     resolve()
//   }, 1000);
// })

// cover_wait.then(() => cover(2))
// cover(1)

let x = 10;

{
  x = 20;
  let y = 30;
}

console.log(x); // 20
// console.log(y); // ReferenceError: y is not defined