let t=setInterval(run,20);
let c = 0;
let i = 0;
let d = Date().toString();
let txt = "Date: " + d + "\nSalut to you!\n";
padd(txt);

function run(){
  let bod = document.getElementById("bod").innerHTML;
  for(let i = 0 ; i < 10; i++){

    let _b = bod.split("\n");
    for (let j = 0; j < _b.length; j++){
      let _l = Array.from(_b[j]);
      let r = Math.floor(Math.random()*_l.length);
      let st = Math.floor(Math.random()*10).toString();
      _l[r] = st;
      _b[j] = _l.join("");
    }
    document.getElementById("bod").innerHTML=_b.join("\n");
    //c++;

    if(c>document.getElementById("bod").innerHTML.length){
    clearInterval(t);
    //  //t=setInterval(next, 20); 
    }

  }
}

function next(){
//  clearInterval(t);
//  i=0;
//  document.getElementById("bod").innerHTML = b.join("");
}

function padd(t){
  for(let j = 0; j < t.length; j++){
    (function(j){
    setTimeout(() => { 
      document.getElementById("top").innerHTML += t.charAt(j);
    }, 20*j);
    }(j));
  }
}

function pp(_in){
  if (i < _in.length) {
    document.getElementById("top").innerHTML += _in.charAt(i);
    i++;
    setTimeout(pp(_in), 40);
  }
}