/*
find predictions
*/
function findTopValues(inp, count) {
    var outp = [];
    let indices = findIndicesOfMax(inp, count)
    // show  scores
    for (var i = 0; i < indices.length; i++)
        outp[i] = inp[indices[i]]
    return outp
}
/*
get indices of the top probs
*/
function findIndicesOfMax(inp, count) {
    var outp = [];
    for (var i = 0; i < inp.length; i++) {
        outp.push(i); // add index to output array
        if (outp.length > count) {
            outp.sort(function(a, b) {
                return inp[b] - inp[a];
            }); // descending sort the output array
            outp.pop(); // remove the last index (index of smallest element in output array)
        }
    }
    return outp;
}
function preprocess(s){
	
}
function predict(s) {
        
        var class_names = ['Male','Female']
        //get the prediction 
        var pred = model.predict(preprocess(s)).dataSync()
        console.log(pred)            
        //retreive the highest probability class label 
        const idx = tf.argMax(pred);

                
        //find the predictions 
        var indices = findIndicesOfMax(pred, 1)
        console.log(indices)
        var probs = findTopValues(pred, 1)
        var names = class_names(indices) 

        //set the table 
        //setTable(names, probs) 
        document.getElementById("Result").innerHTML = names
        
	    console.log(names);
        console.log(document.getElementById("Result"));
    
  }
async function start(){
	//img = document.getElementById('image').files[0];
	
        
        model = await tf.loadModel('model/model.json')
        
        var status = document.getElementById('status')
      
        status.innerHTML = 'Model Loaded'
        
        var myText = document.getElementById("myText");
        var s = myText.value;
        
        predict(s)
         
        }
        