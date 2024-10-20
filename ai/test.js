function test () {
  // let {w, b} = {w:2, b:1}
  let {w, b} = {w:2, b:1}
  const add = () => {
    w += 1;
    b += 1;
  }

  const sum = () => w + b;

  return {w, b, add, sum}
}