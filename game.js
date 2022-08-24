let r=setInterval(run,1000);

let alphabet = "&%*£!)@:;`¬.=[]{}#"

function background(x, y) {
    let buffer = "";
    for(let i=0; i<y;i++){
        for(let j=0;j<x;j++){
            if(Math.random() < 0.2)
                buffer+=alphabet.charAt(Math.floor(Math.random() * alphabet.length));
            else
                buffer+="🮕"; 
        }
        buffer+="\n";
    }
    return buffer;
}

function run(){
    b = background(150, 40)
    document.body.innerHTML=b;
    //clearInterval(r);
}
